from pathlib import Path
import torch
from pytorch_lightning import Trainer
from src.cnnClassifier.components.prepare_base_model import MyImageClassifier
from src.cnnClassifier.components.model_trainer import TrainingDataModule
from src.cnnClassifier.config.configuration import EvaluationConfig
from src.cnnClassifier.utils.common import save_json
from src.cnnClassifier.entity.config_entity import LogConfig
from src.cnnClassifier.config.configuration import ConfigurationManager
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger



class Evaluation:
    def __init__(self, config: EvaluationConfig, logs:LogConfig):
        self.config = config
        self.logs = logs
        self.model = None
        self.datamodule = None
        self.tb_logger = TensorBoardLogger(str(self.logs.tensorboard / "evaluation"))
        self.mlflow_logger = MLFlowLogger(
            experiment_name="ChestCancerClassification_Evaluation",
            tracking_uri=str(self.logs.mlflow)
        )

    def load_model(self):
        checkpoint = torch.load(self.config.path_of_model, map_location="cpu", weights_only=False)
        state_dict = checkpoint['state_dict']
        base_model = torch.load(self.config.base_model_updated, map_location="cpu", weights_only=False)
        model = MyImageClassifier(model=base_model, config=self.config)
        model.load_state_dict(state_dict)
        self.model = model


    def load_data(self):
        self.datamodule = TrainingDataModule(self.config)
        self.datamodule.setup(stage="validate")  # or test

    def evaluation(self):
        self.load_model()
        self.load_data()

        trainer = Trainer(accelerator="cpu", logger=[self.tb_logger, self.mlflow_logger])
        result = trainer.validate(self.model, datamodule=self.datamodule, verbose=True)
        self.score = result[0]
        
        return result
    
    
    def save_score(self):
        scores = {"loss": self.score.get("val_loss"),"accuracy": self.score.get("val_accuracy")}
        save_json(path=Path("scores.json"), data=scores)