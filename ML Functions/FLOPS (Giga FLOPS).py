import tensorflow as tf
import numpy as np

def get_gflops(model, model_inputs) -> float:
    """
    Calculate GFLOPS (Giga Floating Point Operations Per Second) for a tf.keras.Model or
    tf.keras.Sequential model in inference mode. This function utilizes the 
    tf.compat.v1.profiler to estimate the operations under the hood.
    
    Args:
    model (tf.keras.Model | tf.keras.Sequential): The model for which to calculate GFLOPS.
    model_inputs (List[tf.Tensor]): A list of input tensors to the model.
    
    Returns:
    float: The GFLOPS value of the model during inference.
    """
    
    # Check if the model argument is an instance of the supported types.
    if not isinstance(model, (tf.keras.models.Sequential, tf.keras.models.Model)):
        raise ValueError(
            "Calculating GFLOPS is only supported for `tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

    # Set the batch size to 1 for inference mode calculation
    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model_inputs
    ]

    # Convert the Keras model to a TensorFlow concrete function and then to a frozen graph
    # to count FLOPS for operations used during inference.
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Setup the TensorFlow profiler to calculate FLOPS
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    # Calculate FLOPS using the TensorFlow profiler.
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    # Reset the default graph to clean up
    tf.compat.v1.reset_default_graph()

    # Convert total FLOPS to Giga FLOPSN  (GFLOPS) and divide by 2 if needed to adjust the measurement.
    gflops = (flops.total_float_ops / 1e9) / 2

    return gflops

if __name__ == "__main__":
    # Example usage with VGG19
    image_model_eff = tf.keras.applications.VGG19 (include_top=False, weights=None)
    x = tf.constant(np.random.randn(1, 32, 32, 3))
    print(get_gflops(image_model_eff, [x]))
