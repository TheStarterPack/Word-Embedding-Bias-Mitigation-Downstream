import itertools

import datasets
from datasets import load_dataset
from gensim.models import KeyedVectors
from tqdm import tqdm
import tensorflow as tf
from datasets import Dataset
from util import config
from util.config import LM_SEQ_LEN
from util.preprocessing import preprocess_sentence, index_list_of_sentences

LOG_TAG = "LM_DATA"


def get_wiki_text_103_dataset_tokenized(embd: KeyedVectors):
    print("loading WikiText3 dataset")
    train, test, valid = load_dataset('wikitext', 'wikitext-103-raw-v1',
                                      split=[datasets.Split.TRAIN, datasets.Split.TEST, datasets.Split.VALIDATION])
    train_tokenized = preprocess(train, "wiki103-train", embd)
    valid_tokenized = preprocess(valid, "wiki103-valid", embd)

    test_tokenized = preprocess_test(test, embd)

    all_tokens = set()
    for seq in tqdm(train_tokenized + test_tokenized + valid_tokenized, "collecting tokens"):
        for token in seq:
            all_tokens.add(token)

    all_tokens = sorted(list(all_tokens))
    all_tokens_map = {k: v for v, k in enumerate(all_tokens)}

    return reembed_to_dataset(train_tokenized, all_tokens_map, "train"), \
           reembed_to_dataset(valid_tokenized, all_tokens_map, "valid"), \
           reembed_to_test_set(test_tokenized, all_tokens_map), \
           all_tokens


def reembed_to_dataset(tokenized, all_tokens_map, dataset_name):
    data = [[all_tokens_map[i] for i in seq] for seq in tqdm(tokenized, f"re-embedding {dataset_name}")]
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset.map(split_input_target).shuffle(10000).batch(64, drop_remainder=False).prefetch(
        tf.data.experimental.AUTOTUNE)


def reembed_to_test_set(tokenized, all_tokens_map):
    data = [[all_tokens_map[i] for i in seq] for seq in tqdm(tokenized, "re-embedding test set")]
    dataset = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(data))
    return dataset.map(split_input_target).batch(1)


def preprocess(dataset: Dataset, dataset_name, embd):
    sequences = []
    current_sen = []
    for datapoint in tqdm(itertools.islice(dataset, 150000), f"preprocessing {dataset_name} data set"):
        text = datapoint["text"]
        sen = preprocess_sentence(text.strip(), min=LM_SEQ_LEN)
        if text.startswith("=") or text.startswith(" ="):
            if current_sen:
                chunks = [current_sen[x:x + LM_SEQ_LEN] for x in range(0, len(current_sen), LM_SEQ_LEN)]
                last = chunks[-1]
                if len(last) < config.LM_SEQ_MIN_LEN:
                    for chunk in chunks[:-1]:
                        sequences.append(chunk)
                else:
                    while len(last) < LM_SEQ_LEN:
                        last.append(config.PADDING_WORD)
                    sequences.extend(chunks)
            current_sen = []
        elif sen:
            current_sen += sen
    return index_list_of_sentences(sequences, embd, dataset_name)


def preprocess_test(dataset: Dataset, embd: KeyedVectors):
    sequences = []
    for datapoint in tqdm(dataset, "preprocessing test WikiText-103 test data set"):
        text = datapoint["text"]
        if text.startswith("=") or text.startswith(" ="):
            continue
        sen = preprocess_sentence(text.strip())
        if sen is not None:
            sequences.append(text)
    return index_list_of_sentences(sequences, embd, "test")


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text
