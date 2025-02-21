from io import BytesIO

import boto3
import joblib


def load_model_from_s3(bucket_name, model_key):
    s3_client = boto3.client('s3')
    
    # Fetch the model file from S3
    obj = s3_client.get_object(Bucket=bucket_name, Key=model_key)
    model_data = obj['Body'].read()
    
    # Load the model using joblib from the binary data
    model = joblib.load(BytesIO(model_data))
    
    return model
