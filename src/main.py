from DGLim.dataset import AvianNatureSounds, ds
from DGLim.model import *
from DGLim.hyperparameter import hp
from DGLim.train import *

# Parameter 
model = DeepGriffinLim(blocks=10)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=hp.min_lr, verbose=True)

TrainingLoop = ModelTrainer(model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            dataset=ds,
                            batch_size=hp.batch_size,
                            epochs=hp.epochs,
                            learning_rate=hp.learning_rate,
                            save_path='DGLim/checkpoints')

TrainingLoop.main()

