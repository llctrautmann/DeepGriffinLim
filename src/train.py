from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, random_split
from utils import HealthCheckDashboard, send_push_notification, resize_signal_length, visualize_tensor, save_reconstruction
from hyperparameter import hp
import torch
from numpy import inf
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchaudio
import math


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
            subset = Subset(self.dataset, range(20))
            trainset, testset = random_split(subset, [0.9, 0.1])
            trainset, validset = random_split(trainset, [0.9, 0.1])
        else:
            trainset, testset = random_split(self.dataset, [0.9, 0.1])
            trainset, validset = random_split(trainset, [0.9, 0.1])

        # Create dataloaders
        self.train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.val_loader = DataLoader(validset, batch_size=self.batch_size, shuffle=False,pin_memory=True)
        self.test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False,pin_memory=True)

        print(f"Split dataset into {len(trainset)} training samples, {len(validset)} validation samples, and {len(testset)} test samples")
        return trainset, validset, testset

    
    def train(self):
        self.model.train()
        train_loss = 0
        loop = tqdm(self.train_loader, disable=not self.debug)
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
            # loss = self.criterion(z_tilda - clear, residual)


            ### Phase loss, and GD loss and IF loss

            # Create derivatives
            gdl_clear = self.create_derivative(torch.angle(clear), dire='gdl')
            gdl_final = self.create_derivative(torch.angle(final), dire='gdl')

            ifr_clear = self.create_derivative(torch.angle(clear), dire='ifr')
            ifr_final = self.create_derivative(torch.angle(final), dire='ifr')

            # Calculate loss

            if self.loss_type == 'L1':
                loss = self.criterion(z_tilda - clear, residual)
            elif self.loss_type == 'phase':
                loss = self.von_mises_loss(torch.angle(clear), torch.angle(final))
            elif self.loss_type == 'gdl':
                loss = self.von_mises_loss(torch.angle(clear), torch.angle(final)) + self.von_mises_loss(gdl_clear, gdl_final)

            elif self.loss_type == 'ifr':
                loss = self.von_mises_loss(torch.angle(clear), torch.angle(final)) + self.von_mises_loss(ifr_clear, ifr_final)

            elif self.loss_type == 'all':
                loss = self.von_mises_loss(torch.angle(clear), torch.angle(final)) + \
                        self.von_mises_loss(gdl_clear, gdl_final) + \
                        self.von_mises_loss(ifr_clear, ifr_final) 

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update progress bar
            loop.set_description(f"Batch [{idx + 1}/{self.train_loader.__len__()}]")
            loop.set_postfix({'Batch loss': f'{train_loss:.4f}'})

            train_loss += loss.item()

            if idx == 1:
                self.write_to_tensorboard(clear, z_tilda, residual, final)

        return train_loss, final


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

                # Create Phase + Derivatives
                gdl_clear = self.create_derivative(torch.angle(clear), dire='gdl')
                gdl_final = self.create_derivative(torch.angle(final), dire='gdl')

                ifr_clear = self.create_derivative(torch.angle(clear), dire='ifr')
                ifr_final = self.create_derivative(torch.angle(final), dire='ifr')

                # Calculate loss
                if self.loss_type == 'L1':
                    loss = self.criterion(z_tilda - clear, residual)
                elif self.loss_type == 'phase':
                    loss = self.von_mises_loss(torch.angle(clear), torch.angle(final))
                elif self.loss_type == 'gdl':
                    loss = self.von_mises_loss(torch.angle(clear), torch.angle(final)) + self.von_mises_loss(gdl_clear, gdl_final)

                elif self.loss_type == 'ifr':
                    loss = self.von_mises_loss(torch.angle(clear), torch.angle(final)) + self.von_mises_loss(ifr_clear, ifr_final)

                elif self.loss_type == 'all':
                    loss = self.von_mises_loss(torch.angle(clear), torch.angle(final)) + \
                            self.von_mises_loss(gdl_clear, gdl_final) + \
                            self.von_mises_loss(ifr_clear, ifr_final) 

                validation_loss += loss.item()

            
            #save_reconstruction(self.model,step=self.step)
            visualize_tensor(clear,key='clear',step=self.step, loss=self.loss_type)
            visualize_tensor(final,key='final',step=self.step, loss=self.loss_type)

        return validation_loss

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
        print('No checkpoint loaded, weights will be initialised randomly.')

        self.model = self.model.to(self.device)
        validation_losses = []

        loop = tqdm(range(self.epochs), disable=self.debug)
        for epoch in loop:
            self.checkpoint = {'state_dict': self.model.state_dict(),'optimizer': self.optimizer.state_dict(), 'epoch': epoch, 'best_loss': self.best_loss}

            train_loss, final = self.train()
            validation_loss = self.validate()
            validation_losses.append(validation_loss)
            print(f'Epoch: {epoch+1}/{self.epochs} | Train loss: {train_loss:.4f} | Validation loss: {validation_loss:.4f}')

            if not self.debug:
                send_push_notification(epoch, validation_loss)

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

                # Create Phase + Derivatives
                gdl_clear = self.create_derivative(torch.angle(clear), dire='gdl')
                gdl_final = self.create_derivative(torch.angle(final), dire='gdl')

                ifr_clear = self.create_derivative(torch.angle(clear), dire='ifr')
                ifr_final = self.create_derivative(torch.angle(final), dire='ifr')


                # Calculate loss
                if self.loss_type == 'L1':
                    loss = self.criterion(z_tilda - clear, residual)
                elif self.loss_type == 'phase':
                    loss = self.von_mises_loss(torch.angle(clear), torch.angle(final))
                elif self.loss_type == 'gdl':
                    loss = self.von_mises_loss(torch.angle(clear), torch.angle(final)) + self.von_mises_loss(gdl_clear, gdl_final)

                elif self.loss_type == 'ifr':
                    loss = self.von_mises_loss(torch.angle(clear), torch.angle(final)) + self.von_mises_loss(ifr_clear, ifr_final)

                elif self.loss_type == 'all':
                    loss = self.von_mises_loss(torch.angle(clear), torch.angle(final)) + \
                            self.von_mises_loss(gdl_clear, gdl_final) + \
                            self.von_mises_loss(ifr_clear, ifr_final) 

                testing_loss += loss.item()
            self.return_audio_sample(clear=clear,noisy_signal=noisy, final=final)

        return validation_losses


    def saving_checkpoint(self):
        assert isinstance(self.checkpoint, dict), "self.checkpoint must be a dictionary"
        assert isinstance(self.save_path, str), "self.save_path must be a string"
        file_name = f"checkpoint_{self.checkpoint['epoch']}.pth.tar"
        checkpoint_path = os.path.join(self.save_path, file_name)
        print(f"Saving checkpoint at {checkpoint_path}")
        torch.save(self.checkpoint, checkpoint_path)


    def loading_checkpoint(self):
        assert isinstance(self.load_path, str), "self.load_path must be a string"
        if os.path.isfile(self.load_path):
            self.checkpoint = torch.load(self.load_path)
            self.model.load_state_dict(checkpoint)
            if self.verbose:
                print(f"Loaded checkpoint '{self.load_path}'")
                return 1
            return 1
        else:
            print(f"No checkpoint found at '{self.load_path}'")
            return 0

    def von_mises_loss(self, y_true, y_pred):
        """
        Von Mises loss function.

        Args:
        y_true (Tensor): Tensor of true values.
        y_pred (Tensor): Tensor of predicted values.

        Returns:
        Tensor: Loss value.
        """
        # Ensure predictions are in range [-pi, pi]
        y_pred = torch.atan2(torch.sin(y_pred), torch.cos(y_pred))

        # Compute von Mises loss
        loss = 1 - torch.cos(y_pred - y_true)
        return torch.mean(loss)
            

    def write_to_tensorboard(self, clear, z_tilda, residual, final):
        # Convert the tensors to angle before visualizing
        clear_grid = torchvision.utils.make_grid(torch.angle(clear),padding=20)
        residual_grid = torchvision.utils.make_grid(torch.angle(residual),padding=20)
        z_tilda_grid = torchvision.utils.make_grid(torch.angle(z_tilda),padding=20)
        final_grid = torchvision.utils.make_grid(torch.angle(final),padding=20)

        # Add the grid to tensorboard
        self.writer.add_images("Clear", clear_grid, dataformats='CHW', global_step=self.step)
        self.writer.add_images("Residual Maps", residual_grid, dataformats='CHW', global_step=self.step)
        self.writer.add_images("z_tilda", z_tilda_grid, dataformats='CHW', global_step=self.step)
        self.writer.add_images("Final", final_grid, dataformats='CHW', global_step=self.step)
        self.step += 1


    def healthcheck(self):
        dashboard = HealthCheckDashboard(self.train_loader, self.model, self.writer)
        dashboard.perform_healthcheck()

    def return_audio_sample(self, noisy_signal, clear, final, length=hp.length * hp.sampling_rate):
        '''
        Convert clear and reconstruction to audio and save to disk
        '''
        for idx in range(clear.shape[0]):
            sample = clear[idx, ...].cpu().detach()
            path = f'./out/{self.loss_type}/clear_{idx}_type_{self.loss_type}.wav'
            wav = torch.istft(sample, n_fft=hp.n_fft, hop_length=hp.hop_length)
            wav = resize_signal_length(wav, length)
            torchaudio.save(path, wav, hp.sampling_rate//2)

        for idx in range(final.shape[0]):
            sample = final[idx, ...].cpu().detach()
            path = f'./out/{self.loss_type}/recon_{idx}_type_{self.loss_type}.wav'
            wav = torch.istft(sample, n_fft=hp.n_fft, hop_length=hp.hop_length)
            wav = resize_signal_length(wav, length)
            torchaudio.save(path, wav, hp.sampling_rate//2)


        for idx in range(noisy_signal.shape[0]):
            sample = noisy_signal[idx, ...].cpu().detach()
            path = f'./out/{self.loss_type}/noisy_{idx}_type_{self.loss_type}.wav'
            wav = torch.istft(sample, n_fft=hp.n_fft, hop_length=hp.hop_length)
            wav = resize_signal_length(wav, length)
            torchaudio.save(path, wav, hp.sampling_rate//2)
        print('Training complete')

    @staticmethod
    def create_derivative(tensor: torch.Tensor, dire: str, device=hp.device) -> torch.Tensor:
        def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
            # Wrap value to [-pi, pi)
            x += 2 * math.pi * 1e6
            x %= 2 * math.pi
            return torch.where(x >= math.pi, x - 2 * math.pi, x)

        
        if tensor.dim() == 2:
            tensor = tensor.to(device).unsqueeze(0).unsqueeze(0)
        else:
            tensor = tensor.to(device)

        if dire == 'gdl':
            diff = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
            new_matrix = torch.cat([diff, tensor[:, :, -1:, :]], dim=2)
        elif dire == 'ifr':
            if tensor.dim() == 4:
                tensor = tensor.permute(0, 1, 3, 2)
            diff = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
            new_matrix = torch.cat([diff, tensor[:, :, -1:, :]], dim=2)
            new_matrix = new_matrix.permute(0, 1, 3, 2)
        else:
            raise ValueError(f"Unknown direction: {dire}")
        if tensor.dim() == 2:
            return wrap_to_pi(new_matrix).squeeze().squeeze()
        else:
            return wrap_to_pi(new_matrix)
