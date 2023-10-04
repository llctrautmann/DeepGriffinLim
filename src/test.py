from dataset import AvianNatureSounds, ds
from model import *
from hyperparameter import hp
from train import *
from utils import seed_everything
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import random

# Seed everything for reproducibility
# seed_everything()


# def compute_differences(rand_mat):
#     """
#     Compute differences between consecutive columns and rows of a matrix.
    
#     Parameters:
#     - rand_mat (torch.Tensor): Input matrix
    
#     Returns:
#     - if_mat (torch.Tensor): Differences between consecutive columns
#     - gdl_mat (torch.Tensor): Differences between consecutive rows
#     """
    
#     # Compute differences between consecutive columns
#     if_mat = torch.cat([rand_mat[:, 1:] - rand_mat[:, :-1], rand_mat[:, -1:]], dim=1)
    
#     # Compute differences between consecutive rows
#     gdl_mat = torch.cat([-rand_mat[1:, :] + rand_mat[:-1, :], rand_mat[-1:, :]], dim=0)

#     # Wrap the derivatives into the range -pi to pi
#     if_mat = torch.remainder(if_mat + np.pi, 2 * np.pi) - np.pi
#     gdl_mat = torch.remainder(gdl_mat + np.pi, 2 * np.pi) - np.pi
    
#     return if_mat, gdl_mat


def compute_differences(rand_mat):
    """
    Compute differences between consecutive columns and rows of a matrix.
    
    Parameters:
    - rand_mat (torch.Tensor): Input matrix
    
    Returns:
    - if_mat (torch.Tensor): Differences between consecutive columns
    - gdl_mat (torch.Tensor): Differences between consecutive rows
    """
    
    # Compute differences between consecutive columns
    if_mat = torch.cat([rand_mat[:, :, :, 1:] - rand_mat[:, :, :, :-1], rand_mat[:, :, :, -1:]], dim=3)
    
    # Compute differences between consecutive rows
    gdl_mat = torch.cat([-rand_mat[:, :, 1:, :] + rand_mat[:, :, :-1, :], rand_mat[:, :, -1:, :]], dim=2)

    # Wrap the derivatives into the range -pi to pi
    if_mat = torch.remainder(if_mat + np.pi, 2 * np.pi) - np.pi
    gdl_mat = torch.remainder(gdl_mat + np.pi, 2 * np.pi) - np.pi
    
    return if_mat, gdl_mat



if __name__ == '__main__':

    # # Load the sample
    # stft, random_noise, magnitude, label = ds.__getitem__(random.randint(0, len(ds) - 1))

    # # Compute the differences between consecutive columns and rows
    # if_mat, gdl_mat = compute_differences(torch.angle(stft))

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(ds, batch_size=hp.batch_size, shuffle=True, num_workers=hp.num_workers)

    # Iterate over the dataloader
    batch = next(iter(dataloader))

    # Compute the differences between consecutive columns and rows
    
    stft, random_noise, magnitude, label = batch

    for idx, batch in enumerate(dataloader):
        stft, random_noise, magnitude, label = batch

        print("STFT shape: ", stft.shape)
        print("Random noise shape: ", random_noise.shape)
        print("Magnitude shape: ", magnitude.shape)


        if_mat, gdl_mat = compute_differences(torch.angle(stft))
        print(if_mat.shape)
        print(gdl_mat.shape)

        # Plotting the tensors if_mat and gdl_mat
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(if_mat[0][0].numpy(), cmap='hot', interpolation='nearest')
        plt.title('if_mat')

        plt.subplot(1, 2, 2)
        plt.imshow(gdl_mat[0][0].numpy(), cmap='hot', interpolation='nearest')
        plt.title('gdl_mat')

        plt.show()
        break





