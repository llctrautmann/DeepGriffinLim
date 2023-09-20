import librosa
from model import *
import torch
import torchaudio
from utils import *

def plot_spectrograms(batch: torch.Tensor, width=10, height=3):
    plt.figure(figsize=(width, height))
    librosa.display.specshow(batch[0][0].numpy(),
                            sr=44100,
                            x_axis='time',
                            y_axis='linear'
                            )
    plt.colorbar(format="%+2.f")
    plt.show()



model = DeepGriffinLim(blocks=1)
amptodb = torchaudio.transforms.AmplitudeToDB()

file = librosa.load('./data/UK_BIRD/BALMER-01_0_20150621_0515.wav', sr=44100)[0]
file = librosa.load('./data/scale.wav', sr=44100)[0]

stft = torch.stft(torch.from_numpy(file), n_fft=1024, hop_length=512, return_complex=True)


mag = amptodb(torch.abs(stft))
phase = torch.angle(stft)

mag = mag.unsqueeze_(0).unsqueeze_(0)
phase = phase.unsqueeze_(0).unsqueeze_(0)
# plot_spectrograms(phase)

random_phase = torch.rand_like(phase, dtype=torch.complex64)

z_tilda, residual, final, subblock_out = model(x_tilda=random_phase, mag=mag)

print(final.shape)

final_phase = torch.angle(final)
final_mag = torch.abs(final)

plot_spectrograms(final_mag.detach())


