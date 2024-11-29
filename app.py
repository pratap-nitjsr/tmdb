import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from utils import *

# Load the pipeline using pickle
@st.cache(allow_output_mutation=True)
def load_pipeline():
    pipeline_path = "./pipeline.joblib"
    try:
        pipeline = joblib.load(pipeline_path)
        st.write("Pipeline loaded successfully.")
    except Exception as e:
        st.write(f"Failed to load pipeline: {e}")

pipeline = load_pipeline()

# Streamlit App
st.title("CSV File Processor Using Saved Pipeline")
st.write("Upload a CSV file, and the app will process it using the saved pipeline.")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(data.head())

    try:
        # Process the data using the pipeline
        if 'pre_process' in dict(pipeline.named_steps):
            preprocessed_data = pipeline.named_steps['pre_process'].transform(data)
            st.write("### Preprocessed Data")
            st.dataframe(preprocessed_data)

        # Make predictions if the pipeline contains a model
        if 'voting_reg' in dict(pipeline.named_steps):
            predictions = pipeline.predict(data)
            predictions = np.expm1(predictions)
            st.write("### Predictions")
            st.dataframe(pd.DataFrame(predictions, columns=["Predictions"]))
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV file to continue.")
