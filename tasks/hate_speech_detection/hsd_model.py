from keras import Model
from keras.layers import *

from gensim.models import KeyedVectors

from util.embedding_layer import gensim_to_keras_embedding


def get_hsd_model(embds: KeyedVectors) -> Model:
    input = Input(shape=(None,), dtype="int32", ragged=True)
    embedding = gensim_to_keras_embedding(embds)
    x = embedding(input)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, input_shape=(None, 300)))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='tanh')(x)
    x = Dropout(0.5)(x)
    output = Dense(1)(x)

    model = Model(input, output)

    model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model
