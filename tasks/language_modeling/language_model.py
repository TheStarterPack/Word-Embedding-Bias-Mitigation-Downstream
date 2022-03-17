from keras import Model
from keras.layers import *
from keras.losses import SparseCategoricalCrossentropy
from gensim.models import KeyedVectors

from util.embedding_layer import gensim_to_keras_embedding


def get_language_model(embds: KeyedVectors, all_tokens) -> Model:
    input = Input(shape=(None,))

    token_lookup = IntegerLookup(vocabulary=all_tokens, invert=True)
    tokens = token_lookup(input)

    word_embedding = gensim_to_keras_embedding(embds)

    x = word_embedding(tokens)
    x = Dropout(0.7)(x)
    x = LSTM(1500, return_sequences=True, recurrent_dropout=0.7)(x)
    x = LSTM(1500, return_sequences=True, recurrent_dropout=0.7)(x)
    x = Dropout(0.7)(x)
    output = TimeDistributed(Dense(len(all_tokens)))(x)

    model = Model(inputs=input, outputs=output)
    loss = SparseCategoricalCrossentropy(from_logits=True)
    model.compile('adam', loss=loss, metrics=['accuracy'])
    model.summary()

    return model
