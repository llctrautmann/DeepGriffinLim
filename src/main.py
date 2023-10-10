from dataset import AvianNatureSounds, ds
from model import *
from hyperparameter import hp
from train import *
from utils import seed_everything
import wandb

wandb_mode = 'no-sweep'

if hp.device.startswith('cuda'):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Phase Retrieval",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": wandb.config.learning_rate if wandb_mode == 'sweep' else hp.learning_rate,
        "architecture": "Deep Griffin Lim",
        "dataset": "EC_BIRD",
        "epochs": hp.epochs,
        "loss type": hp.loss_type,
        "batch size": wandb.config.batch_size if wandb_mode == 'sweep' else hp.batch_size,
        "phase type": hp.data_mode,
        "weight decay": wandb.config.weight_decay if wandb_mode == 'sweep' else hp.weight_decay,
        }
    )
else:
    pass

# Seed everything for reproducibility
seed_everything()

# Training loop
# Criterion, optimizer, scheduler
model = DeepGriffinLim(blocks=hp.model_depth)
criterion = nn.L1Loss(reduction='sum').to(device=hp.device)
optimizer = torch.optim.Adam(model.parameters(),lr=wandb.config.learning_rate if wandb_mode == 'sweep' else hp.learning_rate, weight_decay=wandb.config.weight_decay if wandb_mode == 'sweep' else hp.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=hp.min_lr, verbose=True)

TrainingLoop = ModelTrainer(model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            dataset=ds,
                            batch_size=wandb.config.batch_size if wandb_mode == 'sweep' else hp.batch_size,
                            epochs=hp.epochs,
                            loss_type=hp.loss_type,
                            learning_rate=wandb.config.learning_rate if wandb_mode == 'sweep' else hp.learning_rate,
                            save_path='./src/checkpoints',
                            debug=False,
                            device=hp.device,
                            load_checkpoint=False,
                            load_path='./src/checkpoints/')

# Training loop Execution
TrainingLoop.main()
