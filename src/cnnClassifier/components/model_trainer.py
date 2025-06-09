import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from src.cnnClassifier.config.configuration import TrainingConfig, LogConfig
from src.cnnClassifier.components.prepare_base_model import MyImageClassifier
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning import Trainer





class TrainingDataModule(pl.LightningDataModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config


    def setup(self, stage=None):
        transform_list = [
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor()
        ]
        if self.config.params_is_augmentation:
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ] + transform_list

        transform = transforms.Compose(transform_list)
        dataset = datasets.ImageFolder(self.config.training_data, transform=transform)

        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.params_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.params_batch_size)
    





class Training:
    def __init__(self, config: TrainingConfig, logs: LogConfig):
        self.config = config
        self.logs = logs
        self.model = None
        self.datamodule = None

    def get_base_model(self):
        base_model = torch.load(self.config.updated_base_model_path, weights_only=False)
        self.model = MyImageClassifier(model=base_model, config=self.config)

    def get_data_module(self):
        self.datamodule = TrainingDataModule(self.config)

    def train(self):
        self.get_base_model()
        self.get_data_module()

        tb_logger = TensorBoardLogger(self.logs.tensorboard)
        mlflow_logger = MLFlowLogger(
            experiment_name="ChestCancerClassification",
            tracking_uri= str(self.logs.mlflow)
            )
        
        trainer = Trainer(
            max_epochs=self.config.params_epochs,
            accelerator="auto",
            logger=[tb_logger, mlflow_logger],
            enable_progress_bar=True,
            log_every_n_steps=1
        )

        trainer.fit(self.model, datamodule=self.datamodule)

        torch.save(self.model.model, self.config.trained_model_path)