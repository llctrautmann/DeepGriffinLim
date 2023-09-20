from dataset import AvianNatureSounds, ds
from model import *
from hyperparameter import hp
from train import *
from utils import seed_everything

# Seed everything for reproducibility
seed_everything()

# Model  
model = DeepGriffinLim(blocks=hp.model_depth)

# Criterion, optimizer, scheduler
criterion = nn.L1Loss(reduction='sum').to(device=hp.device)
optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=hp.min_lr, verbose=True)

# Training loop

loss_types = ['phase']
# loss_types = ['phase', 'gdl', 'ifr', 'all']

for loss_type in loss_types:
    TrainingLoop = ModelTrainer(model=model,
                                criterion=criterion,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                dataset=ds,
                                batch_size=hp.batch_size,
                                epochs=hp.epochs,
                                loss_type=loss_type,
                                learning_rate=hp.learning_rate,
                                save_path='./src/checkpoints',
                                debug=True,
                                device=hp.device)

    # Training loop Execution
    #TrainingLoop.healthcheck()
    TrainingLoop.main()
