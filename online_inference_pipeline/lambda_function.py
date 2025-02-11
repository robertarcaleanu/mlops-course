import json

from model import MyModel

# Load model once at startup to improve performance
model = MyModel()

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))  # Debugging

        # Extract and parse JSON body (for API Gateway Proxy Integration)
        if "body" in event:
            body = json.loads(event["body"])  # Convert string to dictionary
        else:
            body = event  # If it's direct invocation, use as is

        # Ensure 'name' is present
        if "name" not in body:
            raise ValueError("Missing 'name' field in request")

        # Pass parsed JSON to the model
        prediction = model.predict(body)

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": prediction})
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
