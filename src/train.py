from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from numpy import inf
import os

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, dataset, batch_size, epochs, learning_rate,min_lr=5e-8, scheduler=None, patience=10, device='cpu', save_path='./checkpoints/', load_path='./checkpoints/', debug=True, save_checkpoint=True, load_checkpoint=False, verbose=True):
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

        # Checkpoint parameters
        self.checkpoint = None
        self.save_checkpoint = save_checkpoint
        self.load_checkpoint = load_checkpoint
        self.save_path = save_path
        self.load_path = load_path
        self.debug = debug
        self.verbose = verbose
        self.best_loss = inf

        # Init functions
        self.split_datasets()

    def split_datasets(self):
        """
        This function splits the dataset into training, validation, and test sets.
        The dataset is first split into a training set (90% of the data) and a test set (10% of the data).
        The training set is then further split into a training set (90% of the new set) and a validation set (10% of the new set).
        
        Returns:
            trainset (Dataset): The final training set.
            testset (Dataset): The test set.
        """

        if self.debug:
            subset = torch.utils.data.Subset(self.dataset, range(20))


        # trainset, testset = torch.utils.data.random_split(self.dataset, [0.9,0.1])
        trainset, testset = torch.utils.data.random_split(subset, [0.9,0.1])
        trainset, validset = torch.utils.data.random_split(trainset, [0.9, 0.1])

        # Create dataloaders
        self.train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(validset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=True)

        print(f"Split dataset into {len(trainset)} training samples, {len(validset)} validation samples, and {len(testset)} test samples")
        return None

    
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
            loss = self.criterion(z_tilda - clear, residual)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update progress bar
            loop.set_description(f"Batch [{idx + 1}/{self.train_loader.__len__()}]")
            loop.set_postfix({'Batch loss': f'{train_loss:.4f}'})

            train_loss += loss.item()

        return train_loss, final


    def main(self):
        if self.load_checkpoint:
            flag = self.loading_checkpoint()
            if flag == 0:
                print("No checkpoint found")
                print("No training initiated")
                return 0
            else:
                print("Checkpoint loaded")

        validation_losses = []
        for epoch in range(self.epochs):  
            self.checkpoint = {'state_dict': self.model.state_dict(),'optimizer': self.optimizer.state_dict(), 'epoch': epoch, 'best_loss': self.best_loss}

            train_loss, final = self.train()
            validation_loss = self.validate()
            validation_losses.append(validation_loss)

            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                self.saving_checkpoint()

            self.scheduler.step(validation_loss)

            # insert early stopping here
            if  self.optimizer.param_groups[0]['lr'] == self.min_lr:
                if validation_loss > self.best_loss:
                    self.counter += 1
                    print(f'Early stopping counter: {self.counter}')

                    if self.counter == 10:
                        break

                else:
                    self.counter = 0
                    
        print(validation_losses)
        return validation_losses


        # Insert the testing functionality here
        with torch.no_grad():
            self.model.eval()
            testing_loss = 0
            for idx, batch in enumerate(self.test_loader):
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
                loss = self.criterion(z_tilda - clear, residual)
                testing_loss += loss.item()


    def validate(self):
        self.model.eval()
        validation_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
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
                loss = self.criterion(z_tilda - clear, residual)
                validation_loss += loss.item()

        return validation_loss


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


    def write_to_tensorboard(self, writer):
        # TODO: Implement the tensorboard functionality
        pass
    