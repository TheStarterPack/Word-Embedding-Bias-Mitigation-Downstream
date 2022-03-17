from gensim.models import KeyedVectors
from tensorflow import keras


def gensim_to_keras_embedding(model: KeyedVectors, train_embeddings=False):
    weights = model.vectors  # vectors themselves, a 2D numpy array

    layer = keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=train_embeddings,
    )
    return layer
