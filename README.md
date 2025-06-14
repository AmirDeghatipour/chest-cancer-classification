# Chest Cancer Classification

> **⚠️ Disclaimer:**  
> This project is intended for educational purposes only, demonstrating a complete machine learning pipeline.  
> The results are **not reliable for any clinical or real-world predictions**.  
> The model is trained on a limited dataset with minimal hyperparameter tuning and should not be used for diagnostic or decision-making purposes.


This repository contains a deep learning pipeline for classifying chest CT scan images to detect chest cancer using a Convolutional Neural Network (CNN) based on the VGG16 architecture. The project is modular, reproducible, and leverages **PyTorch Lightning**, **DVC**, **MLflow**, and **Streamlit** for experiment tracking and deployment.

## Features

- **Data Ingestion:** Downloads and extracts chest CT scan datasets.
- **Data Versioning and Pipeline Reproducibility:** Uses DVC to manage datasets and ensure reproducible machine learning pipelines.
- **Model Preparation:** Prepares and customizes a VGG16-based model for binary classification.
- **Training:** Trains the model with configurable hyperparameters and supports Optuna-based hyperparameter tuning.
- **Evaluation:** Evaluates the trained model and logs metrics.
- **Experiment Tracking:** Integrated with MLflow and TensorBoard.
- **Deployment:** Provides a Streamlit web app for image-based inference.

## Project Structure 🗂️

```
.
├── config/
│   └── config.yaml
├── artifacts/
├── logs/
├── src/
│   └── cnnClassifier/
│       ├── components/
│       ├── config/
│       ├── constants/
│       ├── entity/
│       ├── logging/
│       ├── pipeline/
│       └── utils/
├── notebooks/
├── main.py
├── streamlit_app.py
├── requirements.txt
├── dvc.yaml
├── params.yaml
├── scores.json
└── README.md
```

## Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/AmirDeghatipour/chest-cancer-classification.git
cd chest-cancer-classification
```

### 2. Install dependencies

It is recommended to use a virtual environment.

```sh
pip install -r requirements.txt
```

### 3. Run the full pipeline with DVC

Instead of manually running each stage, use DVC to execute the full pipeline (data ingestion, model preparation, training, and evaluation) with a single command:

```sh
dvc repro
```

⚠️ Make sure DVC is installed (pip install dvc) and Git is initialized if you're cloning the repo for the first time.

### 4. Experiment Tracking

- **MLflow UI:**  
  Start MLflow UI to track experiments:
  ```sh
  mlflow ui --backend-store-uri ./artifacts/logs/mlruns
  ```
- **TensorBoard:**  
  Start TensorBoard for visualization:
  ```sh
  tensorboard --logdir ./artifacts/logs/tensorboard
  ```

### 5. Model Inference

To run the Streamlit app for image-based inference:

```sh
streamlit run streamlit_app.py
```

Upload a chest CT image to get a prediction.

## Configuration

- **config/config.yaml:** Project and artifact paths.
- **params.yaml:** Model and training hyperparameters.

## Reproducibility

This project uses [DVC](https://dvc.org/) for pipeline reproducibility. You can run the full pipeline with:

```sh
dvc repro
```

## Notebooks

- `notebooks/01_data_ingestion.ipynb`: Data ingestion exploration.
- `notebooks/02_prepare_base_model.ipynb`: Model preparation.
- `notebooks/03_model_trainer.ipynb`: Training and hyperparameter tuning.
- `notebooks/04_model_evaluation.ipynb`: Model evaluation.

## Author

- **Amir Deghatipour**  
  [GitHub](https://github.com/AmirDeghatipour)  
  Email: a.deghatipour@gmail.com


---

> **Note:** For any issues or feature requests, please open an [issue](https://github.com/AmirDeghatipour/chest-cancer-classification/issues).