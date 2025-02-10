class MyModel:
    def __init__(self):
        # Load your trained model here
        pass

    def load_model(self, model_path):
        # Dummy example: Replace with actual model loading code (e.g., TensorFlow, PyTorch)
        print(f"Loading model from {model_path}")
        return None  # Replace with actual model

    def preprocess(self, data):
        # Convert data to model-compatible format
        name = data["name"]
        return f"My name is {name}"

    def predict(self, data):
        # Preprocess input
        processed_data = self.preprocess(data)
        
        return processed_data.upper()
