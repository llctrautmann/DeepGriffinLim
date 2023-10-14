import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import random
import librosa
from hyperparameter import hp
import numpy as np


class AvianNatureSounds(Dataset):
    def __init__(
        self,
        # file args
        annotation_file_path=None,
        root_dir="../",
        key="habitat",
        # sound transformation args
        mode="stft",
        length=5,
        sampling_rate=44100,
        n_fft=1024,
        hop_length=512,
        downsample=True,
        mel_spectrogram=None,
        verbose=False,
    ):
        self.column = key
        self.annotation_file = pd.read_csv(annotation_file_path).sort_values(
            self.column
        )
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
        self.offset_type = 'random'
        self.offset = None
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            power=2,
            n_iter=15,
            momentum=0.99,
        )
        self.pretrained_phases = None
        self.pretrained = []
        self.pretrained_phases = []
        self.original = []
        self.stft = []
        self.magnitude = []
        self.label = []
        if hp.data_mode == 'gla-pretrain':
            self.precompute_gla_phases()

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        if hp.data_mode == "gla-pretrain":
            return torch.from_numpy(self.stft[index]),
            torch.from_numpy(self.pretrained_phases[index]),
            torch.from_numpy(self.magnitude[index]),
            self.label[index]

        if self.mode == "stft":
            audio_sample_path = os.path.join(
                self.root_dir, os.listdir(self.root_dir)[index]
            )
            label = self.annotation_file.iloc[index][self.column]
            # signal, sr = torchaudio.load(audio_sample_path)

            signal, sr = librosa.load(audio_sample_path, sr=self.sampling_rate)
            signal = signal.reshape(1, -1)

            if self.downsample:
                signal = self.downsample_waveform(signal)
            else:
                pass

            # Clip the signal to the desired length
            signal = self.clip(signal, sr, self.length, offset='random')

            # Depreciated STFT function
            # stft = torch.stft(
            #     signal,
            #     n_fft=self.n_fft,
            #     hop_length=self.hop_length,
            #     win_length=self.n_fft,
            #     normalized=False,
            #     return_complex=True,
            # )

            stft = librosa.stft(
                signal,
                n_fft=self.n_fft,
                hop_length=self.hop_length)

            stft = torch.from_numpy(stft)
            # Add complex Gaussian noise to the complex tensor
            noise_real = torch.randn_like(stft.real)
            noise_imag = torch.randn_like(stft.imag)

            noise = torch.complex(noise_real, noise_imag)

            # Compute signal power and noise power
            signal_power = torch.mean(stft.abs().square())
            noise_power = torch.mean(noise.abs().square())

            # Generate a random Signal-to-Noise Ratio (SNR) value between -6 and 0
            random_snr = torch.rand(1).item() * 6 - 6
            # Compute the scaling factor for the noise
            K = torch.sqrt(signal_power / (noise_power * 10 ** (random_snr / 10)))
            # Scale the noise and add it to the original signal
            noisy_sig = stft + K * noise
            magnitude = torch.abs(stft)  # assuming stft is your complex tensor

            # create a random phase
            phase = torch.rand_like(magnitude) * 2 * torch.pi - torch.pi  
            real = magnitude * torch.cos(phase)
            imag = magnitude * torch.sin(phase)
            random_init = torch.complex(real, imag)

            if hp.data_mode == 'denoise':
                return stft, noisy_sig, magnitude, label
            else:
                return stft, random_init, magnitude, label

    def precompute_gla_phases(self):
        for index in range(len(self.annotation_file)):
            audio_sample_path = os.path.join(
                self.root_dir, os.listdir(self.root_dir)[index]
            )

            # Check if the file is a .wav file
            if not audio_sample_path.endswith('.wav'):
                continue
            signal, sr = librosa.load(audio_sample_path, sr=self.sampling_rate)
            signal_length = signal.shape[0]
            signal = signal.reshape(1, -1)
            if self.downsample:
                signal = self.downsample_waveform(signal)
            signal = self.clip(signal, sr, self.length, offset='random')
            stft = librosa.stft(
                signal,
                n_fft=self.n_fft,
                hop_length=self.hop_length)

            magnitude = np.abs(stft)
            self.magnitude.append(magnitude)

            self.stft.append(stft)
            self.label.append(self.annotation_file.iloc[index][self.column])

            self.original.append(signal)
            gla_pretrained = librosa.griffinlim(magnitude,n_iter=1, n_fft=1024, hop_length=512,length=signal.shape[1])
            gla_pretrained_phase = np.angle(librosa.stft(gla_pretrained, n_fft=1024, hop_length=512))

            self.pretrained.append(gla_pretrained)
            self.pretrained_phases.append(gla_pretrained_phase)

    @torch.no_grad()
    def clip(self, audio_signal, sr, desired_length, offset=None):
        """
        Clips an audio signal to a desired length.

        Args:
        audio_signal (Tensor): Tensor of shape (..., time) representing the waveform to be clipped.
        sr (int): Sampling rate of the audio signal.
        desired_length (float): Desired length of the audio signal in seconds.
        offset (int, optional): Starting point of the clip in the audio signal. If None, a random offset is chosen. If 'start', the clip starts from the beginning of the audio signal.

        Returns:
        Tensor: Clipped audio signal.
        """
        sig_len = audio_signal.shape[1]
        length = int(sr * desired_length)
        if sig_len > length:
            if self.offset_type == 'random':
                if self.offset is None:
                    # offset = random.randint(0, sig_len - length)
                    offset = 10000
                    self.offset = offset
                else:
                    offset = self.offset
                audio_signal = audio_signal[:, offset : (offset + length)]
                return audio_signal
            else:
                audio_signal = audio_signal[:, self.offset : (self.offset + length)]
                return audio_signal

        elif offset == 'start':
            audio_signal = audio_signal[:, :length]
            return audio_signal
        elif offset == 'all':
            return audio_signal

    @staticmethod
    @torch.no_grad()
    def downsample_waveform(waveform, orig_freq=48000, new_freq=16000):
        """
        Downsamples a PyTorch tensor representing a waveform.

        Args:
        waveform (Tensor): Tensor of shape (..., time) representing the waveform to be resampled.
        orig_freq (int, optional): Original frequency of the waveform. Defaults to 44100.
        new_freq (int, optional): Frequency to downsample to. Defaults to 16000.

        Returns:
        Tensor: Downsampled waveform.
        """
        transform = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=new_freq, lowpass_filter_width=128
        )
        return transform(waveform)


ds = AvianNatureSounds(
    annotation_file_path=hp.annotation_file_path,
    root_dir=hp.root_dir,
    key=hp.key,
    mode=hp.mode,
    length=hp.length,
    sampling_rate=hp.sampling_rate,
    n_fft=hp.n_fft,
    hop_length=hp.hop_length,
    mel_spectrogram=hp.mel_spectrogram,
    downsample=hp.downsample,
)
