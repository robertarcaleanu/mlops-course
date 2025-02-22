# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11

# Copy model and Lambda function
COPY . .

# Install dependencies (if needed)
RUN pip install --no-cache-dir -r requirements.txt

# Set the Lambda handler function
CMD ["lambda_function.lambda_handler"]
