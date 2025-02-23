import json
import logging

from online_inference_pipeline.model import inference
from online_inference_pipeline.transform import Transformer, preprocess
from online_inference_pipeline.utils import load_object_from_s3


def lambda_handler(event, context):
    try:
        logging.info(f"Event received {event}")
        df = preprocess(event)
        logging.info("Event Preprocessed")
        df = Transformer().transform(df)
        logging.info("Event Transformed")

        model = load_object_from_s3(
            bucket_name="dataset-mlops-robert", model_key="model_dag.joblib"
        )
        logging.info("model loaded")
        prediction = inference(df, model)
        logging.info("Prediction completed")

        return {"statusCode": 200, "body": json.dumps({"prediction": prediction}), "version": "@v1"}
    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}


# Local testing code
# if __name__ == "__main__":
#     with open('online_inference_pipeline/input.json') as f:
#         event = json.load(f)  # Load the event from the file
#     context = {}  # You can leave the context empty for local testing
#     response = lambda_handler(event, context)
#     print("Response:", response)
