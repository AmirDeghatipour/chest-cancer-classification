import torch
from torchvision import transforms
from PIL import Image
from PIL import Image
from pathlib import Path
from src.cnnClassifier.components.prepare_base_model import MyImageClassifier
from src.cnnClassifier.entity.config_entity import EvaluationConfig





class PredictionPipeline:
    def __init__(self, filename, config: EvaluationConfig):
        self.filename = filename
        self.config = config


    def predict(self):

        
        checkpoint = torch.load(self.config.path_of_model, map_location="cpu", weights_only=False)
        state_dict = checkpoint['state_dict']
        base_model = torch.load(self.config.base_model_updated, map_location="cpu", weights_only=False)
        model = MyImageClassifier(model=base_model, config=self.config)
        model.load_state_dict(state_dict)


        model.eval()


        imagename = self.filename
        img = Image.open(imagename).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0) # type: ignore
        img_tensor = img_tensor.to("cpu")


        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            result = predicted.item()


        if result == 1:
            prediction = 'It has Chest Cancer'
        else:
            prediction = 'No Chest Cancer'

        return [{"image": prediction}]