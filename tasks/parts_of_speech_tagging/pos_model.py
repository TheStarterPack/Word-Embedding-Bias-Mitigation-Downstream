from keras import Model
from keras.layers import *
import keras
from gensim.models import KeyedVectors

from util import config
from util.embedding_layer import gensim_to_keras_embedding


def get_pos_model(embds: KeyedVectors) -> Model:
    input = Input(shape=(None,))
    embedding = gensim_to_keras_embedding(embds)
    x = embedding(input)
    x = Dropout(0.7)(x)
    x = Bidirectional(LSTM(512, return_sequences=True))(x)
    x = Dropout(0.7)(x)
    output = TimeDistributed(Dense(config.POS_CLASSES))(x)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = Model(inputs=input, outputs=output)
    model.compile("adam", loss=loss, metrics=["accuracy"])

    model.summary()
    return model
