import datasets
import numpy as np
import tensorflow as tf
import tqdm
from datasets import load_dataset
from gensim.models import KeyedVectors

from util import config
from util.preprocessing import get_token_index


def get_conll2003_dataset(emb: KeyedVectors, target):
    print("Loading CoNLL-2003 Dataset")
    if target not in ["ner", "pos"]:
        raise ValueError("please choose valid target: ner or pos")
    train, valid = load_dataset("conll2003", split=[datasets.Split.TRAIN, datasets.Split.VALIDATION])
    return process_dataset(train, emb, "conll2003-train", target), \
           process_dataset(valid, emb, "conll2003-valid", target)


def get_conll2003_test(emb: KeyedVectors, target):
    if target not in ["ner", "pos"]:
        raise ValueError("please choose valid target: ner or pos")
    test = load_dataset("conll2003", split=datasets.Split.TEST)
    X = []
    y = []
    for datapoint in tqdm.tqdm(test, f"processing test"):
        tokens = datapoint["tokens"]
        labels = datapoint[f"{target}_tags"]
        filtered_tokens = []
        filtered_tags = []
        for i, token in enumerate(tokens):
            if token.isalnum():
                filtered_tags.append(labels[i])
                filtered_tokens.append(token.lower())
        tokens = [get_token_index(emb, token) for token in filtered_tokens]
        if len(tokens) >= 1:
            X.append(np.asarray(tokens))
            y.append(np.asarray(filtered_tags))
    return X, y


def process_dataset(dataset, embedding: KeyedVectors, dataset_name, target):
    X = []
    y = []
    for datapoint in tqdm.tqdm(dataset, f"processing {dataset_name}"):
        tokens = datapoint["tokens"]
        labels = datapoint[f"{target}_tags"]
        filtered_tokens = []
        filtered_tags = []
        for i, token in enumerate(tokens):
            if token.isalnum():
                filtered_tags.append(labels[i])
                filtered_tokens.append(token.lower())

        if len(filtered_tokens) < config.CONLL2003_MIN_SEN_LEN:
            continue
        if len(filtered_tokens) > config.CONLL2003_SEN_LEN:
            filtered_tokens = filtered_tokens[:config.CONLL2003_SEN_LEN]
            filtered_tags = filtered_tags[:config.CONLL2003_SEN_LEN]
        while len(filtered_tokens) < config.CONLL2003_SEN_LEN:
            if target == "ner":
                filtered_tokens.append(config.PADDING_WORD)
                filtered_tags.append(9)
            elif target == "pos":
                filtered_tokens.append(config.PADDING_WORD)
                filtered_tags.append(47)

        tokens = [get_token_index(embedding, token) for token in filtered_tokens]
        X.append(np.asarray(tokens))
        y.append(np.asarray(filtered_tags))

    return tf.data.Dataset.from_tensor_slices((X, y)).shuffle(1000).batch(32, drop_remainder=False).prefetch(
        tf.data.experimental.AUTOTUNE)
