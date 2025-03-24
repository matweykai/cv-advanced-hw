from model import YOLO
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint


trainer = Trainer(
    max_epochs=10,
    callbacks=[ModelCheckpoint(save_top_k=1, mode="max", monitor="val_loss")],
    logger=CSVLogger("logs"),
)

model = YOLO(num_classes=2, num_anchors=3, num_features=3)
trainer = Trainer()
trainer.fit(model)