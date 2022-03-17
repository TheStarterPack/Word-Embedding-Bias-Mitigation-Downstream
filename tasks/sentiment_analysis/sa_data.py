import datasets
from datasets import load_dataset

import tensorflow as tf

from util.preprocessing import preprocess_sentence, index_list_of_sentences
from sklearn.model_selection import train_test_split


def get_imdb_dataset_tokenized(emb):
    print("Loading IMDB Review Dataset")
    train, test = load_dataset("imdb", split=[datasets.Split.TRAIN, datasets.Split.TEST])
    review_set = set()

    train_X, train_Y = [], []
    test_X, test_Y = [], []

    for datapoint in train:
        text = datapoint["text"]
        if text not in review_set:
            review_set.add(text)
            train_X.append(text)
            train_Y.append(datapoint["label"])

    for datapoint in test:
        text = datapoint["text"]
        if text not in review_set:
            review_set.add(text)
            test_X.append(text)
            test_Y.append(datapoint["label"])

    test_X, valid_X, test_Y, valid_Y = train_test_split(test_X, test_Y, test_size=0.5, shuffle=True, random_state=23845,
                                                        stratify=test_Y)

    return preprocess_dataset(train_X, train_Y, emb, "imdb_train"), \
           preprocess_dataset(test_X, test_Y, emb, "imdb_test"), \
           preprocess_dataset(valid_X, valid_Y, emb, "imdb_valid")


def preprocess_dataset(texts, labels, emb, name):
    y = []
    sentences = []
    for i, sen in enumerate(texts):
        sen_list = preprocess_sentence(sen)
        if sen_list is not None:
            y.append(labels[i])
            sentences.append(sen_list)
    X = index_list_of_sentences(sentences, emb, name)
    return tf.data.Dataset.from_tensor_slices((tf.ragged.constant(X), tf.ragged.constant(y))).shuffle(25000).batch(32)
