import argparse
from functools import partial
from itertools import chain
import json
import logging
import os
from pathlib import Path
import random
import time
from dataclasses import dataclass, field
from typing import Tuple, Union
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
import pandas as pd

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from fastprogress import master_bar, progress_bar
from omegaconf import OmegaConf, open_dict, MISSING
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from networks.convrnn_embedder import ConvRNNConfig, ConvRNNEmbedder
from networks.ge2e_loss import GE2ELoss
from dataset import UtteranceDS, SpecialCollater

@dataclass
class DistributedConfig:
    dist_backend: str = 'nccl'
    dist_url: str = "tcp://localhost:54321"
    n_nodes: int = 1
    n_gpus_per_node: int = 1

@dataclass
class TrainConfig:
    # Distributed settings
    distributed: DistributedConfig = DistributedConfig()
    # Model settings
    model_cfg: ConvRNNConfig = ConvRNNConfig()
    n_uttr_per_spk: int = 6

    device: str = 'cuda'
    seed: int = 1775
    
    batch_size: int = 8
    num_workers: int = 6
    fp16: bool = False
    n_epochs: int = 50
    summary_interval: int = 25
    checkpoint_interval: int = 2500
    stdout_interval: int = 100
    validation_interval: int = 1000

    # Learning settings
    start_lr: float = 1e-6
    max_lr: float = 5e-5
    end_lr: float = 3e-7
    warmup_pct: float = 0.15
    betas: Tuple[float, float] = (0.9, 0.99)
    grad_clip: float = 1.0

    # Data settings
    checkpoint_path: str = MISSING
    train_csv: str = MISSING
    valid_csv: str = MISSING
    resume_checkpoint: str = ''
    sample_rate: int = 16000
    min_seq_len: int = 16000*2
    max_seq_len: int = 16000*8
    
def lin_one_cycle(startlr, maxlr, endlr, warmup_pct, total_iters, iters):
    """ 
    Linearly warms up from `startlr` to `maxlr` for `warmup_pct` fraction of `total_iters`, 
    and then linearly anneals down to `endlr` until the final iter.
    """
    warmup_iters = int(warmup_pct*total_iters)
    if iters < warmup_iters:
        # Warmup part
        m = (maxlr - startlr)/warmup_iters
        return m*iters + startlr
    else:
        m = (endlr - maxlr)/(total_iters - warmup_iters)
        c = endlr - total_iters*m
        return m*iters + c    

def flatten_cfg(cfg: Union[DictConfig, ListConfig]) -> dict:
    """ 
    Recursively flattens a config into a flat dictionary compatible with 
    tensorboard's `add_hparams` function.
    """
    out_dict = {}
    if type(cfg) == ListConfig:
        cfg = DictConfig({f"[{i}]": v for i, v in enumerate(cfg)})

    for key in cfg:
        if type(getattr(cfg, key)) in (int, str, bool, float):
            out_dict[key] = getattr(cfg, key)
        elif type(getattr(cfg, key)) in [DictConfig, ListConfig]:
            out_dict = out_dict | {f"{key}{'.' if type(getattr(cfg, key)) == DictConfig else ''}{k}": v for k, v in flatten_cfg(getattr(cfg, key)).items()}
        else: raise AssertionError
    return out_dict

def train(rank, cfg: TrainConfig):
    if cfg.distributed.n_gpus_per_node > 1:
        init_process_group(backend=cfg.distributed.dist_backend, init_method=cfg.distributed.dist_url,
                           world_size=cfg.distributed.n_nodes*cfg.distributed.n_gpus_per_node, rank=rank)

    device = torch.device(f'cuda:{rank:d}')

    model = ConvRNNEmbedder(cfg.model_cfg).to(device)
    loss_fn = GE2ELoss(device).to(device)
    
    logging.info(f"Initialized rank {rank}")
    
    if rank == 0:
        logging.getLogger().setLevel(logging.INFO)
        logging.info(f"Model initialized as:\n {model}")
        os.makedirs(cfg.checkpoint_path, exist_ok=True)
        logging.info(f"checkpoints directory : {cfg.checkpoint_path}")
        logging.info(f"Model has {sum([p.numel() for p in model.parameters()]):,d} parameters.")

    steps = 0
    if cfg.resume_checkpoint != '' and os.path.isfile(cfg.resume_checkpoint):
        state_dict = torch.load(cfg.resume_checkpoint, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        loss_fn.load_state_dict(state_dict['loss_fn_state_dict'])
        steps = state_dict['steps'] + 1
        last_epoch = state_dict['epoch']
        print(f"Checkpoint loaded from {cfg.resume_checkpoint}. Resuming training from {steps} steps at epoch {last_epoch}")
    else:
        state_dict = None
        last_epoch = -1

    if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1:
        if rank == 0: logging.info("Multi-gpu detected")
        model = DDP(model, device_ids=[rank]).to(device)
        loss_fn = DDP(loss_fn, device_ids=[rank]).to(device)

    optim = torch.optim.AdamW(chain(model.parameters(), loss_fn.parameters()), 1.0, betas=cfg.betas)
    if state_dict is not None: optim.load_state_dict(state_dict['optim_state_dict'])

    train_df, valid_df = pd.read_csv(cfg.train_csv), pd.read_csv(cfg.valid_csv)

    trainset = UtteranceDS(train_df, cfg.sample_rate, cfg.n_uttr_per_spk)

    train_sampler = DistributedSampler(trainset) if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else None

    train_loader = DataLoader(trainset, num_workers=cfg.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=cfg.batch_size,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=SpecialCollater(cfg.min_seq_len, cfg.max_seq_len))

    if rank == 0:
        validset = UtteranceDS(valid_df, cfg.sample_rate, cfg.n_uttr_per_spk)
        validation_loader = DataLoader(validset, num_workers=cfg.num_workers, shuffle=False,
                                       sampler=None,
                                       batch_size=cfg.batch_size,
                                       pin_memory=False,
                                       drop_last=True,
                                       collate_fn=SpecialCollater(cfg.min_seq_len, cfg.max_seq_len))

        sw = SummaryWriter(os.path.join(cfg.checkpoint_path, 'logs'))

    total_iters = cfg.n_epochs*len(train_loader)
    def sched_lam(x):
        return lin_one_cycle(cfg.start_lr, cfg.max_lr, cfg.end_lr, 
                        cfg.warmup_pct, total_iters, x)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=[sched_lam], last_epoch=steps-1)

    if state_dict is not None:
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    if cfg.fp16: 
        scaler = GradScaler()
        if state_dict is not None and 'scaler_state_dict' in state_dict:
            scaler.load_state_dict(state_dict['scaler_state_dict'])

    model.train()
    
    if rank == 0: 
        mb = master_bar(range(max(0, last_epoch), cfg.n_epochs))
        smooth_loss = None
    else: mb = range(max(0, last_epoch), cfg.n_epochs)

    for epoch in mb:
        if rank == 0:
            start = time.time()
            mb.write("Epoch: {}".format(epoch+1))

        if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1:
            train_sampler.set_epoch(epoch)

        if rank == 0: pb = progress_bar(enumerate(train_loader), total=len(train_loader), parent=mb)
        else: pb = enumerate(train_loader)
        
        for i, batch in pb:
            if rank == 0: start_b = time.time()
            x, xlen = batch
            x = x.to(device, non_blocking=True)
            xlen = xlen.to(device, non_blocking=True)
            
            optim.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                embeds = model(x, xlen)
                loss = loss_fn(embeds)
            if cfg.fp16: 
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                gnorm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                torch.nn.utils.clip_grad.clip_grad_norm_(loss_fn.parameters(), cfg.grad_clip/2)
                scaler.step(optim)
                scaler.update()
            else: 
                loss.backward()
                gnorm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                torch.nn.utils.clip_grad.clip_grad_norm_(loss_fn.parameters(), cfg.grad_clip/2)
                optim.step()

            if rank == 0:
                if smooth_loss is None: smooth_loss = float(loss.item())
                else: smooth_loss = smooth_loss + 0.1*(float(loss.item()) - smooth_loss)
                # STDOUT logging
                if steps % cfg.stdout_interval == 0:
                    mb.write('steps : {:,d}, loss : {:4.3f}, sec/batch : {:4.3f}, peak mem: {:5.2f}GB'. \
                            format(steps, loss.item(), time.time() - start_b, torch.cuda.max_memory_allocated()/1e9))
                    mb.child.comment = 'steps : {:,d}, loss : {:4.3f}, sec/batch : {:4.3f}'. \
                            format(steps, loss.item(), time.time() - start_b)     
                    # mb.write(f"lr = {float(optim.param_groups[0]['lr'])}")            

                # checkpointing
                if steps % cfg.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = f"{cfg.checkpoint_path}/ckpt_{steps:08d}.pt"
                    torch.save({
                        'model_state_dict': (model.module if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else model).state_dict(),
                        'loss_fn_state_dict': (loss_fn.module if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else loss_fn).state_dict(),
                        'optim_state_dict': optim.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': (scaler.state_dict() if cfg.fp16 else None),
                        'steps': steps,
                        'epoch': epoch
                    }, checkpoint_path)
                    logging.info(f"Saved checkpoint to {checkpoint_path}")

                # Tensorboard summary logging
                if steps % cfg.summary_interval == 0:
                    sw.add_scalar("training/loss_smooth", smooth_loss, steps)
                    sw.add_scalar("training/loss_raw", loss.item(), steps)
                    sw.add_scalar("ge2e/w", float((loss_fn.module if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else loss_fn).w.item()), steps)
                    sw.add_scalar("ge2e/b", float((loss_fn.module if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else loss_fn).b.item()), steps)
                    sw.add_scalar("opt/lr", float(optim.param_groups[0]['lr']), steps)
                    sw.add_scalar('opt/grad_norm', float(gnorm), steps)

                # Validation
                if steps % cfg.validation_interval == 0 and steps != 0:
                    model.eval()
                    loss_fn.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    flat_embeds = []
                    flat_lbls = []
                    with torch.no_grad():
                        for j, batch in progress_bar(enumerate(validation_loader), total=len(validation_loader), parent=mb):
                            x, xlen = batch
                            embeds = model(x.to(device), xlen.to(device))
                            val_err_tot += loss_fn(embeds)

                            if j <= 2:
                                lbls = [f'spk-{j}-{indr:03d}' for indr in range(cfg.batch_size) for _ in range(cfg.n_uttr_per_spk)]
                                fembeds = embeds.view(cfg.batch_size*cfg.n_uttr_per_spk, cfg.model_cfg.fc_dim)
                                flat_embeds.append(fembeds.cpu())
                                flat_lbls.extend(lbls)
                            elif j == 3:
                                flat_embeds = torch.cat(flat_embeds, dim=0)
                                sw.add_embedding(flat_embeds, metadata=flat_lbls, global_step=steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/loss", val_err, steps)
                        mb.write(f"validation run complete at {steps:,d} steps. validation loss: {val_err:5.4f}")

                    model.train()
                    loss_fn.train()
                    sw.add_scalar("memory/max_allocated_gb", torch.cuda.max_memory_allocated()/1e9, steps)
                    sw.add_scalar("memory/max_reserved_gb", torch.cuda.max_memory_reserved()/1e9, steps)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()

            steps += 1
            scheduler.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
    sw.add_hparams(flatten_cfg(cfg), metric_dict={'validation/loss': val_err}, run_name=f'run-{cfg.checkpoint_path}')
    print("Training completed!")


def main():
    print('Initializing Training Process..')
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(usage='\n' + '-'*10 + ' Default config ' + '-'*10 + '\n' + 
                            str(OmegaConf.to_yaml(OmegaConf.structured(TrainConfig))))
    a = parser.parse_known_args()
    override_cfg = OmegaConf.from_cli()
    base_cfg = OmegaConf.structured(TrainConfig)
    cfg: TrainConfig = OmegaConf.merge(base_cfg, override_cfg)
    logging.info(f"Running with config:\n {OmegaConf.to_yaml(cfg)}")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        if cfg.distributed.n_gpus_per_node > torch.cuda.device_count():
            raise AssertionError((f" Specified n_gpus_per_node ({cfg.distributed.n_gpus_per_node})"
                                    f" must be less than or equal to cuda device count ({torch.cuda.device_count()}) "))
        with open_dict(cfg):
            cfg.batch_size_per_gpu = int(cfg.batch_size / cfg.distributed.n_gpus_per_node)
        if cfg.batch_size % cfg.distributed.n_gpus_per_node != 0:
            logging.warn(("Batch size does not evenly divide among GPUs in a node. "
                            "Likely unbalanced loads will occur."))
        logging.info(f'Batch size per GPU : {cfg.batch_size_per_gpu}')

    if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1:
       mp.spawn(train, nprocs=cfg.distributed.n_gpus_per_node, args=(cfg,))
    else:
       train(0, cfg)


if __name__ == '__main__':
    main()
