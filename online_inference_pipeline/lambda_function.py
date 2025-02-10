import json

from model import MyModel

# Load model once at startup to improve performance
model = MyModel()

def lambda_handler(event, context):
    try:
        # Parse input from API Gateway event
        # body = json.loads(event.get("body", "{}"))  # Extract JSON body
        prediction = model.predict(event)

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": prediction})
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
