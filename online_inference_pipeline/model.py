import logging


def inference(df, model):
    """
    Perform inference using the trained model.
    """
    logging.info("Performing inference with the model.")
    
    try:
        # Predict using the model
        result = model.predict(df)
        logging.info(f"Inference completed. Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}.")
    
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise
    
    return result
