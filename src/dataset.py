import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import random
from hyperparameter import hp

class AvianNatureSounds(Dataset):
    def __init__(self,
                # file args
                annotation_file_path=None,
                root_dir='../',
                key='habitat',

                # sound transformation args
                mode='stft',
                length=5,
                sampling_rate=44100,
                n_fft=1024,
                hop_length=512,
                downsample=True,
                mel_spectrogram = None,
                verbose=False,
                ):
        
        self.column = key
        self.annotation_file = pd.read_csv(annotation_file_path).sort_values(self.column)
        self.root_dir = root_dir
        self.mel_transformation = mel_spectrogram
        self.AmplitudeToDB = torchaudio.transforms.AmplitudeToDB()
        self.mode = mode
        self.length = length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.downsample = downsample
        self.sampling_rate = sampling_rate
        self.signal_length = None
        self.griffin_lim = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, 
                                                            win_length=self.n_fft,
                                                            hop_length=self.hop_length,
                                                            power=2,
                                                            n_iter=5,
                                                            momentum=0.99)

    def __len__(self):
        return len(self.annotation_file)
    

    def __getitem__(self, index):
        if self.mode == 'stft':
            audio_sample_path = os.path.join(self.root_dir,os.listdir(self.root_dir)[index])
            label = self.annotation_file.iloc[index][self.column]
            signal, sr = torchaudio.load(audio_sample_path)

            if self.downsample:
                signal = self.downsample_waveform(signal)
            else:
                pass

            # Clip the signal to the desired length
            signal = self.clip(signal, sr, self.length)
            # print(f'{signal.shape} = clipped signal shape')

            stft = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length,win_length=self.n_fft, normalized=False, return_complex=True)

            # Add complex Gaussian noise to the complex tensor
            noise_real = torch.randn_like(stft.real)
            noise_imag = torch.randn_like(stft.imag)

            noise = torch.complex(noise_real, noise_imag)

            # Compute signal power and noise power
            signal_power = torch.mean(stft.abs().square())
            noise_power = torch.mean(noise.abs().square())

            random_snr = torch.rand(1).item() * 6 - 6     
             # Compute the scaling factor for the noise
            K = torch.sqrt(signal_power / (noise_power * 10**(random_snr / 10)))
            
            # Scale the noise and add it to the original signal
            noisy_sig = stft + K * noise
            
            # noisy_sig = stft + (noise_real + 1j * noise_imag)
            magnitude = torch.abs(stft) # 25 Jul 2023 @ 12:21:38 ### CHANGED ###

            return stft, noisy_sig, magnitude, label


    @staticmethod
    @torch.no_grad()
    def clip(audio_signal, sr, desired_length):
        sig_len = audio_signal.shape[1]
        length = int(sr * desired_length)
        if sig_len > length:
            offset = random.randint(0, sig_len - length)
            audio_signal = audio_signal[:, offset:(offset+length)]
            return audio_signal
        else:
            return audio_signal
    
    @staticmethod
    @torch.no_grad()
    def downsample_waveform(waveform, orig_freq=44100, new_freq=16000):
        """
        Downsamples a PyTorch tensor representing a waveform.

        Args:
        waveform (Tensor): Tensor of shape (..., time) representing the waveform to be resampled.
        orig_freq (int, optional): Original frequency of the waveform. Defaults to 44100.
        new_freq (int, optional): Frequency to downsample to. Defaults to 16000.

        Returns:
        Tensor: Downsampled waveform.
        """
        transform = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq, lowpass_filter_width=128)
        return transform(waveform)


ds = AvianNatureSounds(annotation_file_path=hp.annotation_file_path,
                       root_dir=hp.root_dir,
                       key=hp.key,
                       mode=hp.mode,
                       length=hp.length,
                       sampling_rate=hp.sampling_rate,
                       n_fft=hp.n_fft,
                       hop_length=hp.hop_length,
                       mel_spectrogram=hp.mel_spectrogram)
