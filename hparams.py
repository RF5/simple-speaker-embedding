
class hp():
        
    class audio_waveglow():
        filter_length = 1024
        hop_length = 256
        win_length = 1024
        n_mel_channels = 80
        sampling_rate = 22050
        mel_fmin = 0
        mel_fmax = 8000.0
        max_wav_value = 32768.0
        min_log_value = -11.52
        max_log_value = 1.2
        silence_threshold_db = -10
        
hparams = hp