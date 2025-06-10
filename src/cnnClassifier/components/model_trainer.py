import os
import glob
import shutil
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from src.cnnClassifier.config.configuration import TrainingConfig, LogConfig
from src.cnnClassifier.components.prepare_base_model import MyImageClassifier
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from copy import deepcopy
import optuna
import mlflow





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
        return DataLoader(self.train_dataset, batch_size=self.config.params_batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.params_batch_size, num_workers=4, persistent_workers=True)
    





class Training:
    def __init__(self, config: TrainingConfig, logs: LogConfig):
        self.config = config
        self.logs = logs
        self.model = None
        self.datamodule = None
        self.best_val_loss_overall = float('inf')

    def get_base_model(self):
        base_model = torch.load(self.config.updated_base_model_path, weights_only=False)
        self.model = MyImageClassifier(model=base_model, config=self.config)

    def get_data_module(self):
        self.datamodule = TrainingDataModule(self.config)


    def get_loggers(self):
        tb_logger = TensorBoardLogger(self.logs.tensorboard)
        mlflow_logger = MLFlowLogger(
            experiment_name="ChestCancerClassification",
            tracking_uri= str(self.logs.mlflow)
            )
        
        return [tb_logger, mlflow_logger]
    


    def get_checkpoint_callback(self):
        checkpoint_dir = getattr(self.config, "checkpoint_dir", self.config.root_dir)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
            dirpath=checkpoint_dir,
            filename="best-model-{epoch:02d}-{val_loss:.4f}"
        )

        return checkpoint_callback



    def train(self):
        self.get_base_model()
        self.get_data_module()


        trainer = Trainer(
            max_epochs=self.config.params_epochs,
            accelerator="auto",
            logger=self.get_loggers(),
            callbacks=[self.get_checkpoint_callback()],
            enable_progress_bar=True,
            log_every_n_steps=1
        )

        trainer.fit(self.model, datamodule=self.datamodule)
        best_model_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                best_model_path = cb.best_model_path
                break
        val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(1.0)).item()
        print(f"Best model saved at: {best_model_path}")


        return val_loss, best_model_path
    



    def suggest_hyperparameters(self, trial):
        config_new = deepcopy(self.config)
        config_new.params_learning_rate = trial.suggest_float("LEARNING_RATE", 1e-5, 1e-1)
        config_new.params_batch_size = trial.suggest_categorical("BATCH_SIZE", [8, 16, 32])

        trial_ckpt_dir = os.path.join(self.config.root_dir, f"checkpoints_trial_{trial.number}")
        os.makedirs(trial_ckpt_dir, exist_ok=True)
        config_new.checkpoint_dir = trial_ckpt_dir

        return config_new


    def save_best_model_if_needed(self, val_loss, best_model_path):
        if val_loss < self.best_val_loss_overall:
            self.best_val_loss_overall = val_loss
            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
            shutil.copy(best_model_path, self.config.trained_model_path)
            print(f"New best model saved with val_loss={val_loss:.4f} at {self.config.trained_model_path}")




    def objective(self,trial):
        
        mlflow.set_tracking_uri(str(self.logs.mlflow))
        config_dict = self.suggest_hyperparameters(trial)

        with mlflow.start_run():
            trainer = Training(config_dict, self.logs)
            val_loss, best_model_path = trainer.train()
            self.save_best_model_if_needed(val_loss, best_model_path)
            print("\n\n")
            return val_loss
        


    def run_optuna_study(self, n_trials: int = 5):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)

        print("Best trial:")
        print(f"  Value: {study.best_trial.value}")
        print(f"  Params: {study.best_trial.params}")
        print(f"Best model checkpoint can be found in: {self.config.trained_model_path}")

        self.cleanup_trial_checkpoints()



    def cleanup_trial_checkpoints(self):
        print("Cleaning up temporary trial checkpoint directories...")
        trial_dirs = glob.glob(os.path.join(self.config.root_dir, "checkpoints_trial_*"))
        final_dir = os.path.dirname(self.config.trained_model_path)
        for d in trial_dirs:
            if os.path.abspath(d) != os.path.abspath(final_dir):
                shutil.rmtree(d, ignore_errors=True)
        print("Cleanup complete. Only the best model directory remains.")