import torchaudio
import torch
import librosa 
import numpy as np


def signal_process(audio_samples, sr=22050, method='logspectrogram'):
    """
    :param audio_samples: sampled raw audio batched inputs (torch.Tensor)
    :param sr: sampling rate
    :param method: signal process methods
    :return: signal_processed output [batch_size, feature_size, sequence_length]
    """    
    
    # TODO: define your signal process method with various functions and hyper parameters
    if method == 'own_way':
        print('!! define your signal process method with various functions and hyper parameters !!')
        raise NotImplementedError
        
    elif method == 'raw_audio':
        return audio_samples
    
    elif method == 'spectrogram':
        spec_layer = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=1024, normalized=True)
        return spec_layer(audio_samples)
    
    elif method == 'logspectrogram':
        spec_layer = torchaudio.transforms.Spectrogram(n_fft=2048, win_length=2048, hop_length=1379, normalized=True)
        spec = spec_layer(audio_samples)
        log_spec= torch.log(torch.clamp(spec, min=1e-6))
        return log_spec
    
    elif method == 'melspectrogram':
        mel_layer = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=1024, normalized=True)
        return mel_layer(audio_samples)
    
    elif method == 'logmelspectrogram':
        mel_layer = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=1379, normalized=True, n_mels=1024)
        mel_spec = mel_layer(audio_samples)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-6))
        return log_mel
    
    elif method == 'mfcc':
        melkwargs = {'hop_length': 1024, 'n_mels': 128}
        mfcc_layer = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40, melkwargs=melkwargs)
        return mfcc_layer(audio_samples)
    
    elif method == 'stft':
        stft= torch.stft(audio_samples, n_fft=2048, hop_length=1024, window=torch.hann_window(2048))
        stft= torch.sqrt(torch.pow(stft[:,:,:,0], 2) + torch.pow(stft[:,:,:,1], 2))
        return stft
    
    elif method == 'librosa_stft':
        audio_samples = audio_samples.numpy()
        f = list()
        for i in range(len(audio_samples)):
            f.append(np.abs(librosa.stft(audio_samples[i], n_fft=2048, hop_length=1024)))
        return torch.Tensor(f)
    
    elif method == 'librosa_cqt':
        audio_samples = audio_samples.numpy()
        f = list()
        for i in range(len(audio_samples)):
            f.append(np.abs(librosa.cqt(audio_samples[i], sr=float(sr), hop_length=1024, bins_per_octave=12, n_bins=7*12, tuning=None)))
        return torch.Tensor(f)
    
    elif method == 'librosa_mfcc':
        audio_samples = audio_samples.numpy()
        f = list()
        for i in range(len(audio_samples)):
            f.append(librosa.feature.mfcc(audio_samples[i], n_mfcc=40, hop_length=1024, n_mels=128))
        return torch.Tensor(f)
    
    elif method == 'librosa_melspectrogram':
        audio_samples = audio_samples.numpy()
        f = list()
        for i in range(len(audio_samples)):
            f.append(librosa.feature.melspectrogram(audio_samples[i], sr=sr, n_fft=2048, hop_length=1024, n_mels=128))
        return torch.Tensor(f)
    
    elif method == 'librosa_logmelspectrogram':
        audio_samples = audio_samples.numpy()
        f = list()
        for i in range(len(audio_samples)):
            f.append(librosa.core.power_to_db(librosa.feature.melspectrogram(audio_samples[i], sr=sr, n_fft=2048, hop_length=1024, n_mels=128)))
        return torch.Tensor(f)
    
    elif method == 'librosa_chroma_cqt':
        audio_samples = audio_samples.numpy()
        f = list()
        for i in range(len(audio_samples)):
            # cqt = np.abs(librosa.cqt(audio_samples[i], sr=float(sr), hop_length=2760, bins_per_octave=24, n_bins=4*24, tuning=None))
            # cqt = np.abs(librosa.cqt(audio_samples[i], sr=float(sr), hop_length=2048, bins_per_octave=12, n_bins=7*12, tuning=None))
            # f.append(librosa.feature.chroma_cqt(C=cqt, n_chroma=24, bins_per_octave=24, n_octaves=7))
            # cqt = np.abs(librosa.cqt(audio_samples[i], sr=float(sr), hop_length=2048, bins_per_octave=12, n_bins=7*12, tuning=None))
            f.append(np.abs(librosa.feature.chroma_cqt(audio_samples[i], sr=float(sr), hop_length=1024, fmin=100, n_chroma=96, n_octaves=4, bins_per_octave=96)))
        return torch.Tensor(f)

    elif method == 'librosa_chroma_cens':
        audio_samples = audio_samples.numpy()
        f = list()
        for i in range(len(audio_samples)):
            cqt = np.abs(librosa.cqt(audio_samples[i], sr=float(sr), hop_length=1024, bins_per_octave=12, n_bins=7*12, tuning=None))
            f.append(librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7))
        return torch.Tensor(f)

    elif method == 'librosa_rms':
        audio_samples = audio_samples.numpy()
        f = list()
        for i in range(len(audio_samples)):
            stft = np.abs(librosa.stft(audio_samples[i], n_fft=2048, hop_length=1024))
            f.append(librosa.feature.rms(S=stft))
        return torch.Tensor(f)
    
    else:
        raise NotImplementedError
        
        
if __name__ == '__main__':
    y, sr = librosa.load(librosa.util.example_audio_file() , mono=True)
    y = torch.Tensor(y).unsqueeze(0)

    print(signal_process(y, method='raw_audio').shape)
    print(signal_process(y, method='spectrogram').shape)
    print(signal_process(y, method='logspectrogram').shape)
    print(signal_process(y, method='melspectrogram').shape)
    print(signal_process(y, method='logmelspectrogram').shape)
    print(signal_process(y, method='mfcc').shape)
    print(signal_process(y, method='stft').shape)

    print(signal_process(y, method='librosa_stft').shape)
    print(signal_process(y, method='librosa_cqt').shape)
    print(signal_process(y, method='librosa_mfcc').shape)
    print(signal_process(y, method='librosa_melspectrogram').shape)
    print(signal_process(y, method='librosa_logmelspectrogram').shape)
    print(signal_process(y, method='librosa_chroma_cqt').shape)
    print(signal_process(y, method='librosa_chroma_cens').shape)
    print(signal_process(y, method='librosa_rms').shape)
