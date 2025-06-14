# streamlit_app.py

import streamlit as st
from src.cnnClassifier.pipeline.prediction import PredictionPipeline
from src.cnnClassifier.config.configuration import ConfigurationManager
import tempfile


st.set_page_config(page_title="Chest Cancer Classification", layout="centered")
st.title("Chest Cancer Classification")


uploaded_file = st.file_uploader("Please Upload a Chest CT", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Picture", use_container_width=True)


    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_image_path = temp_file.name

    config = ConfigurationManager()
    eval_config = config.get_evaluation_config()
    pipeline = PredictionPipeline(temp_image_path, eval_config)
    result = pipeline.predict()

    st.success(f"Model Result: {result[0]['image']}")