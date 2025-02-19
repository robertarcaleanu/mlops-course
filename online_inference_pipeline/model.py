import logging
from io import BytesIO

import boto3
import joblib
import pandas as pd

from online_inference_pipeline.transform import Transformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyModel:
    def __init__(self):
        # Initialize the transformer and model path (optional, can be passed later)
        self.transformer = Transformer()
        logger.info("MyModel initialized with transformer.")

    def online_inference(self, data):
        logger.info("Received input for inference.")
        
        # Preprocess API input
        preprocessed_data = self._preprocess_api_input(data)
        logger.info(f"Preprocessing completed. Data shape: {preprocessed_data.shape} (if applicable).")
        
        # Transform data using the transformer
        df = self.transformer.transform(preprocessed_data)
        logger.info(f"Transformation completed. Transformed data shape: {df.shape}.")

        # Load the trained model
        model = self._load_model(model_path='s3://your-bucket-name/your-model.joblib')
        
        # Make a prediction
        result = self._inference(df, model)
        logger.info("Inference completed.")

        return {"result": result.tolist()}  # Convert result to a list if it's a numpy array

    def _load_model(self, model_path):
        """
        Load the model from S3 using joblib.
        """
        logger.info(f"Loading model from {model_path}...")
        
        try:
            # S3 client setup (ensure AWS credentials are configured)
            s3_client = boto3.client('s3')
            
            # Extract bucket and key from model_path
            bucket_name, key = model_path.replace("s3://", "").split("/", 1)
            
            # Get the model file from S3
            logger.info(f"Retrieving model from S3 bucket '{bucket_name}' with key '{key}'.")
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            
            # Load the model with joblib
            model = joblib.load(BytesIO(response['Body'].read()))
            logger.info("Model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading model from S3: {e}")
            raise
        
        return model

    def _preprocess_api_input(self, data):
        """
        Convert raw API data into a format compatible with the model.
        """
        logger.info("Preprocessing input data.")
        
        try:
            # Assuming 'data' is a dictionary-like object; modify as needed
            df = pd.DataFrame(data)  # Transform the input JSON data into a DataFrame
            logger.info(f"Input data transformed into DataFrame. Shape: {df.shape}.")
            
        except Exception as e:
            logger.error(f"Error preprocessing input data: {e}")
            raise
        
        return df

    def _inference(self, df, model):
        """
        Perform inference using the trained model.
        """
        logger.info("Performing inference with the model.")
        
        try:
            # Predict using the model
            result = model.predict(df)
            logger.info(f"Inference completed. Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}.")
        
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
        
        return result
