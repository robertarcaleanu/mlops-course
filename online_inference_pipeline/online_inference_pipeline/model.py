import logging


def inference(df, model):
    """
    Perform inference using the trained model.
    """
    logging.info("Performing inference with the model.")

    try:
        # Predict using the model
        result = model.predict(df)
        logging.info(
            f"Inference completed. Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}."
        )
        if result[0] == 1:
            return "yes"

        elif result[0] == 0:
            return "no"

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise
