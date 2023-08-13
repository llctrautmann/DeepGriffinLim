import os
import torch
import shutil
import librosa
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from IPython.display import Audio


def visualize_complex_tensor(tensor):
    if tensor.isinstance(torch.Tensor):
        tensor = tensor.detach()
    else:
        pass
    # Separate the real and imaginary parts of the complex tensor
    real = tensor.abs()
    imag = tensor.angle()
    amptodb = torchaudio.transforms.AmplitudeToDB()

    # Create a grid of subplots for real and imaginary parts
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the real part
    im1 = axs[0].imshow(amptodb(real), cmap='magma')
    axs[0].set_title('Mag')

    # Plot the imaginary part
    im2 = axs[1].imshow(imag, cmap='Blues')
    axs[1].set_title('Phase')

    # Add color bar
    fig.colorbar(im1, ax=axs[0])
    fig.colorbar(im2, ax=axs[1])

    # Show the plot
    plt.show()


def plot_spectrograms(batch: torch.Tensor,magnitude=True,width=10,height=3):
    plt.figure(figsize=(width,height))
    if magnitude:
        librosa.display.specshow(batch[0][0].numpy(),
                                sr=48000,
                                x_axis='time',
                                y_axis='linear',
                                # vmin=0,
                                # vmax=44100//2
                                )
        plt.colorbar(format="%+2.f")
        plt.show()
    else:
        librosa.display.specshow(batch[0][1].numpy(),
                                sr=48000,
                                x_axis='time',
                                y_axis='linear',
                                cmap='Blues',
                                )
        plt.colorbar(format="%+2.f")
        plt.show()


def play_tensor(wave: torch.tensor,sr=44100):
    numpy_waveform = wave.numpy()
    return Audio(numpy_waveform, rate=sr)


def plot_to_tensorboard(writer, loss_critic, loss_gen,real,fake, tb_step,images=False):
    writer.add_scalar('loss_critic', loss_critic, tb_step)
    writer.add_scalar('loss_gen', loss_gen, tb_step)


    if images:

        with torch.no_grad():
            img_grid_fake = torchvision.utils.make_grid(fake[:4], normalize=True) # added [1] so we plot the reconstructed phase not magnitude
            img_grid_real = torchvision.utils.make_grid(real[:4], normalize=True)

            writer.add_image('img_grid_fake', img_grid_fake, tb_step)
            writer.add_image('img_grid_real', img_grid_real, tb_step)


def clear_folders(*folders):
    try:
        # Clear each folder
        for folder in folders:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        print("Content of the folders cleared successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")


