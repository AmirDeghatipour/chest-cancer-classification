import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchinfo import summary
from src.cnnClassifier.config.configuration import PrepareBaseModelConfig




class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        super().__init__()
        self.config = config


    def get_base_model(self):
        self.base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        if not self.config.params_include_top:
            self.base_model.classifier = nn.Identity()

        self.save_model(path=self.config.base_model_path, model=self.base_model)



    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for idx, child in enumerate(model.features):
                if idx < freeze_till:
                    for param in child.parameters():
                        param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, classes),
            nn.Softmax(dim=1)
        )
        
        summary(model,input_size=(1, 3, 224, 224))
        return model



    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.base_model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
        )
        self.save_model(self.config.updated_base_model_path, self.full_model)

    @staticmethod
    def save_model(path, model):
        torch.save(model, path)





class MyImageClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, config: PrepareBaseModelConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=['model'])

    
    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.params_learning_rate)
        return optimizer