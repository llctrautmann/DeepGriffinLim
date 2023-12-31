from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, random_split
from utils import HealthCheckDashboard, send_push_notification, resize_signal_length, visualize_tensor
from hyperparameter import hp
import torch
from numpy import inf
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchaudio
import math
import librosa
import matplotlib.pyplot as plt
import wandb

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, dataset, batch_size, epochs, learning_rate, loss_type='phase', min_lr=5e-8, scheduler=None, patience=10, device='cpu', save_path='./checkpoints/', load_path='./checkpoints/', debug=True, save_checkpoint=True, load_checkpoint=False, verbose=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler is not None else torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=patience,min_lr=min_lr, verbose=True)

        # Device
        self.device = device

        # Train loader
        self.dataset = dataset
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Training parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.loss_type = loss_type


        # Checkpoint parameters
        self.checkpoint = None
        self.save_checkpoint = save_checkpoint
        self.load_checkpoint = load_checkpoint
        self.save_path = save_path
        self.load_path = load_path
        self.debug = debug
        self.verbose = verbose
        self.best_loss = inf


        # Tensorboard
        self.writer = SummaryWriter(f'./tests/runs/debug_run')
        self.step = 0
        
        # Init functions
        self.split_datasets()
            
    def split_datasets(self):
        """
        Returns:
            trainset (Dataset): The final training set.
            testset (Dataset): The test set.
        """

        if self.debug:
            subset = Subset(self.dataset, range(hp.subset_size))
            trainset, testset = random_split(subset, [0.9, 0.1])
            trainset, validset = random_split(trainset, [0.9, 0.1])
        else:
            trainset, testset = random_split(self.dataset, [0.9, 0.1])
            trainset, validset = random_split(trainset, [0.9, 0.1])


        # Create dataloaders
        self.train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.val_loader = DataLoader(validset, batch_size=self.batch_size, shuffle=False,pin_memory=True)
        self.test_loader = DataLoader(testset, batch_size=1, shuffle=False,pin_memory=True)

        print(f"Split dataset into {len(trainset)} training samples, {len(validset)} validation samples, and {len(testset)} test samples")

        return trainset, validset, testset

    
    def train(self):
        self.model.train()
        train_loss = 0
        loop = tqdm(self.train_loader, disable=True)
        for idx, batch in enumerate(loop):
            clear, noisy, mag, label = batch

            # Transfer batch to device
            clear = clear.to(self.device)

            # Transfer batch to device
            noisy = noisy.to(self.device)

            # Transfer batch to device
            mag = mag.to(self.device)

            # Forward pass
            z_tilda, residual, final, subblock_out = self.model(x_tilda=noisy, mag=mag)

            # Calculate loss
            loss, gdl_clear, gdl_final, ifr_clear, ifr_final = self.compute_loss(z_tilda, clear, residual, final)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update progress bar
            loop.set_description(f"Batch [{idx + 1}/{self.train_loader.__len__()}]")

            train_loss += loss.item()

        return train_loss / len(self.train_loader), final


    def validate(self):
        self.model.eval()
        validation_loss = 0
        with torch.no_grad():
            for idx, val_batch in enumerate(self.val_loader):
                clear, noisy, mag, label = val_batch
                # Transfer batch to device
                clear = clear.to(self.device)

                # Transfer batch to device
                noisy = noisy.to(self.device)

                # Transfer batch to device
                mag = mag.to(self.device)

                # Forward pass
                z_tilda, residual, final, subblock_out = self.model(x_tilda=noisy, mag=mag)

                # Calculate loss
                loss, gdl_clear, gdl_final, ifr_clear, ifr_final = self.compute_loss(z_tilda, clear, residual, final)
                validation_loss += loss.item()

                if idx == 0:
                    self.plot_phases(orientation='horizontal',
                    epoch=self.step,
                    loss=self.loss_type,
                    include_phase=True,
                    stft=clear.detach().cpu(),
                    final_stft=final.detach().cpu(),
                    if_mat=ifr_final.detach().cpu(),
                    gdl_mat=gdl_final.detach().cpu()
                    )

            visualize_tensor(clear=clear,recon=final,key='Summary',step=self.step, loss=self.loss_type)
        return validation_loss / len(self.val_loader)

    def main(self):
        print(f'Initialising model on {self.device}.')
        if self.load_checkpoint:
            flag = self.loading_checkpoint()
            if flag == 0:
                print("No checkpoint found, with self.load_checkpoint=True")
                print("No training initiated")
                return 0
            else:
                print("Checkpoint loaded")
        else:
            print('No checkpoint loaded, weights will be initialised randomly.')

        self.model = self.model.to(self.device)
        loop = tqdm(range(self.epochs), disable=self.debug)
        for epoch in loop:
            self.step += 1
            self.checkpoint = {'state_dict': self.model.state_dict(),'optimizer': self.optimizer.state_dict(), 'epoch': epoch, 'best_loss': self.best_loss}
            train_loss, final = self.train()
            validation_loss = self.validate()
            if hp.device.startswith(hp.wandb_device):
                wandb.log({"Training Loss": train_loss, "Validation Loss": validation_loss}, step=self.step)
            else:
                pass
        
            if not self.debug:
                send_push_notification(epoch=epoch, loss_type=self.loss_type, message=validation_loss)
            else:
                pass

            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                self.saving_checkpoint()

            self.scheduler.step(validation_loss)

            # insert early stopping here
            if self.optimizer.param_groups[0]['lr'] == self.min_lr:
                if validation_loss > self.best_loss:
                    self.counter += 1
                    print(f'Early stopping counter: {self.counter}')

                    if self.counter == 10:
                        break
                else:
                    self.counter = 0
                    
        with torch.no_grad():
            self.model.eval()
            testing_loss = 0
            sample_no = 0
            for batch in self.test_loader:
                clear, noisy, mag, label = batch
                # Transfer batch to device
                clear = clear.to(self.device)

                # Transfer batch to device
                noisy = noisy.to(self.device)

                # Transfer batch to device
                mag = mag.to(self.device)

                # Forward pass
                z_tilda, residual, final, subblock_out = self.model(x_tilda=noisy, mag=mag)

                # Calculate loss
                loss, gdl_clear, gdl_final, ifr_clear, ifr_final = self.compute_loss(z_tilda, clear, residual, final)

                testing_loss += loss.item()
                self.return_audio_sample(clear=clear,noisy_signal=noisy, final=final, sample_no=sample_no)
                sample_no += 1
        return testing_loss


    def compute_loss(self, z_tilda, clear, residual, final):
        gdl_clear, ifr_clear = self.create_derivative(torch.angle(clear))
        gdl_final, ifr_final = self.create_derivative(torch.angle(final))
        
        if self.loss_type == 'L1':
            return self.criterion(z_tilda - clear, residual), gdl_clear, gdl_final, ifr_clear, ifr_final
        elif self.loss_type == 'phase':
            return self.von_mises_loss(y_true=torch.angle(clear), y_pred=torch.angle(final), kappa=1.0), gdl_clear, gdl_final, ifr_clear, ifr_final


        elif self.loss_type == 'gdl':
            return self.von_mises_loss(y_true=torch.angle(clear), y_pred=torch.angle(final), kappa=1.0) + self.von_mises_loss(y_true=gdl_clear, y_pred=gdl_final, kappa=1.0), gdl_clear, gdl_final, ifr_clear, ifr_final
        elif self.loss_type == 'ifr':
            return self.von_mises_loss(y_true=torch.angle(clear), y_pred=torch.angle(final), kappa=1.0) + self.von_mises_loss(y_true=ifr_clear, y_pred=ifr_final, kappa=1.0), gdl_clear, gdl_final, ifr_clear, ifr_final
        elif self.loss_type == 'all':
            return self.von_mises_loss(y_true=torch.angle(clear), y_pred=torch.angle(final), kappa=1.0) + \
                self.von_mises_loss(y_true=gdl_clear, y_pred=gdl_final, kappa=1.0) + \
                self.von_mises_loss(y_true=ifr_clear, y_pred=ifr_final, kappa=1.0), gdl_clear, gdl_final, ifr_clear, ifr_final
        elif self.loss_type == 'all_l1':
            return self.criterion(z_tilda - clear, residual) + \
                self.von_mises_loss(y_true=torch.angle(clear), y_pred=torch.angle(final), kappa=1.0) + \
                self.von_mises_loss(y_true=gdl_clear, y_pred=gdl_final, kappa=1.0) + \
                self.von_mises_loss(y_true=ifr_clear, y_pred=ifr_final, kappa=1.0), gdl_clear, gdl_final, ifr_clear, ifr_final
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def saving_checkpoint(self):
        assert isinstance(self.checkpoint, dict), "self.checkpoint must be a dictionary"
        assert isinstance(self.save_path, str), "self.save_path must be a string"
        file_name = f"type_{self.loss_type}_checkpoint_{self.checkpoint['epoch']}.pth.tar"
        checkpoint_path = os.path.join(self.save_path, file_name)
        print(f"Saving checkpoint at {checkpoint_path}")
        torch.save(self.checkpoint, checkpoint_path)

    def loading_checkpoint(self):
        assert isinstance(self.load_path, str), "self.load_path must be a string"
        if os.path.isfile(self.load_path):
            self.checkpoint = torch.load(self.load_path, map_location=torch.device('cpu'))
            
            # Extract the model's state_dict from the checkpoint
            model_state_dict = self.checkpoint.get('state_dict', self.checkpoint)
            
            # Load the model's state_dict
            self.model.load_state_dict(model_state_dict)
            
            if self.verbose:
                print(f"Loaded checkpoint '{self.load_path}'")
                return 1
            return 1
        else:
            print(f"No checkpoint found at '{self.load_path}'")
            return 0

    def von_mises_loss(self, y_true, y_pred, kappa=1.0):
        """
        Von Mises loss function.

        Args:
        y_true (Tensor): Tensor of true values.
        y_pred (Tensor): Tensor of predicted values.
        kappa (float): Concentration parameter.

        Returns:
        Tensor: Loss value.
        """
        return torch.sum(1.0 - torch.exp(kappa * torch.cos(y_true - y_pred)))
            
    def healthcheck(self):
        dashboard = HealthCheckDashboard(self.train_loader, self.model, self.writer)
        dashboard.perform_healthcheck()

    def return_audio_sample(self, noisy_signal, clear, final, length=hp.length * hp.sampling_rate,sample_no=0):
        '''
        Convert clear and reconstruction to audio and save to disk
        '''
        for idx in range(clear.shape[0]):
            sample = clear[idx, ...].cpu().detach()
            path = f'../out/Sample_no_{sample_no}_{self.loss_type}/clear_{idx}_type_{self.loss_type}.wav'
            wav = torch.istft(sample, n_fft=hp.n_fft, hop_length=hp.hop_length)
            wav = resize_signal_length(wav, length)
            wandb.log({f"audio clear {idx}": wandb.Audio(wav.detach().numpy().reshape(-1), caption=f"Clear_{idx}", sample_rate=hp.sampling_rate//2)})

        for idx in range(final.shape[0]):
            sample = final[idx, ...].cpu().detach()
            path = f'../out/Sample_no_{sample_no}_{self.loss_type}/recon_{idx}_type_{self.loss_type}.wav'
            wav = torch.istft(sample, n_fft=hp.n_fft, hop_length=hp.hop_length)
            wav = resize_signal_length(wav, length)
            wandb.log({f"audio recon {idx}": wandb.Audio(wav.detach().numpy().reshape(-1), caption=f"Recon_{idx}", sample_rate=hp.sampling_rate//2)})

        for idx in range(noisy_signal.shape[0]):
            sample = noisy_signal[idx, ...].cpu().detach()
            path = f'../Sample_no_{sample_no}_out/{self.loss_type}/noisy_{idx}_type_{self.loss_type}.wav'
            wav = torch.istft(sample, n_fft=hp.n_fft, hop_length=hp.hop_length)
            wav = resize_signal_length(wav, length)
            wandb.log({f"audio noisy {idx}": wandb.Audio(wav.detach().numpy().reshape(-1), caption=f"Noisy_{idx}", sample_rate=hp.sampling_rate//2)})
        print('Training complete')
    
    def plot_phases(self, orientation='horizontal', epoch=None, loss=None, include_phase=True, stft=None, final_stft=None, if_mat=None, gdl_mat=None):
        if orientation == 'vertical':
            fig, axs = plt.subplots(2, 2, figsize=(15, 15)) if include_phase else plt.subplots(3, 1, figsize=(15, 20))
        elif orientation == 'horizontal':
            fig, axs = plt.subplots(2, 2, figsize=(20, 10)) if include_phase else plt.subplots(1, 3, figsize=(20, 5))
        else:
            raise ValueError("Invalid orientation. Choose either 'vertical' or 'horizontal'")

        mag = np.abs(stft)
        mag_to_db = librosa.amplitude_to_db(mag)
        conv_phase = torch.angle(final_stft)

        axs[0][0].set_title('Magnitude to dB')
        librosa.display.specshow(mag_to_db[0][0], ax=axs[0][0], y_axis='log')
        fig.colorbar(axs[0][0].collections[0], ax=axs[0][0])

        axs[0][1].set_title('GDL Matrix')
        librosa.display.specshow(gdl_mat[0][0].cpu().numpy(), ax=axs[0][1], y_axis='linear')
        fig.colorbar(axs[0][1].collections[0], ax=axs[0][1])

        axs[1][0].set_title('IF Matrix')
        librosa.display.specshow(if_mat[0][0].cpu().numpy(), ax=axs[1][0],y_axis='linear')
        fig.colorbar(axs[1][0].collections[0], ax=axs[1][0])

        if include_phase:
            axs[1][1].set_title('Phase')
            librosa.display.specshow(conv_phase[0][0].cpu().numpy(), ax=axs[1][1], y_axis='linear')
            fig.colorbar(axs[1][1].collections[0], ax=axs[1][1])

        plt.tight_layout()
        img_path = f'../out/img/{epoch}_{loss}.png'
        print(f'Saving image to {img_path}')
        plt.savefig(img_path)
        plt.close()

        # Log the image to wandb
        if hp.device.startswith(hp.wandb_device):
            wandb.log({"Phases": [wandb.Image(img_path, caption=f"Epoch: {epoch}, Loss: {loss}")]})

    @staticmethod
    def create_derivative(rand_mat):
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

