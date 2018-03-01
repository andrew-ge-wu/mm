import cloudpickle
import pickle
import os.path
from .keras_classifier import KerasClassifier


def save_trained(pipeline, directory):
    """
    Save the model pipeline.

    This method saves the pre-processing part as a pickle file and
    the actual Keras classifier in the h5 format and
    a TensorFlow graph to use with TensorFlow Serving.

    Args:
        pipeline: The full pipeline including pre-processing steps and a classifier.
        directory: The directory to save the model
    """

    # Save the Keras classifier for TensorFlow Serving.
    # This needs to be done prior to saving other parts of the pipeline
    # because saved_model_builder will refuse to write into an existing directory.
    import keras.backend as K

    keras_model = pipeline.pipes[-1].model

    # all new operations will be in test mode from now on
    K.set_learning_phase(0)

    from tensorflow.python.saved_model import builder as saved_model_builder
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

    builder = saved_model_builder.SavedModelBuilder(directory)
    signature = predict_signature_def(inputs={'data': keras_model.input},
                                      outputs={'scores': keras_model.output})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={'predict': signature})
        builder.save()

        # Save the Keras classifier in the h5 format.
        keras_model.save(directory + "/keras_model.h5")

    # Save the pre-processing part to a pickle file.
    with open(directory + "/pipeline.pkl", 'wb') as f:
        # Pickle the data using the highest protocol available.
        cloudpickle.dump(pipeline, f, pickle.HIGHEST_PROTOCOL)


def load_trained(directory):
    """
    Load the model pipeline.

    This method loads the pre-processing part from a pickle file and
    the actual Keras classifier from an h5 file.

    Args:
        directory: The directory to load the model from

    Returns:
        Pipeline object

    """
    import keras.backend as K
    import tensorflow as tf
    from keras.models import load_model

    # Load the pre-processing part from a pickle file.
    with open(directory + "/pipeline.pkl", 'rb') as f:
        pipeline = cloudpickle.load(f)

    sess = tf.Session()
    K.set_session(sess)
    model = load_model(directory + "/keras_model.h5")

    # set the model to the pipeline
    pipeline.pipes[-1].model = model

    return pipeline


def save_untrained(pipeline, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    if isinstance(pipeline.pipes[-1], KerasClassifier):
        keras_model = pipeline.pipes[-1].model
        json_string = keras_model.to_json()

        with open(directory + '/keras_achitecture.json', 'w') as f:
            f.write(json_string)

    with open(directory + "/pipeline.pkl", 'wb') as f:
        # Pickle the data using the highest protocol available.
        cloudpickle.dump(pipeline, f, pickle.HIGHEST_PROTOCOL)


def load_untrained(directory):
    with open(directory + "/pipeline.pkl", 'rb') as f:
        pipeline = cloudpickle.load(f)

    if isinstance(pipeline.pipes[-1], KerasClassifier):
        from keras.models import model_from_json

        with open(directory + '/keras_achitecture.json', 'r') as f:
            keras_model = model_from_json(f.read())

        # set the model to the pipeline
        pipeline.pipes[-1].model = keras_model
        pipeline.pipes[-1].compile_model()

    return pipeline
