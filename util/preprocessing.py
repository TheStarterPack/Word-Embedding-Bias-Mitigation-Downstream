import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm
import nltk.tokenize

nltk.download('punkt')


def preprocess_sentence(sen: str, min=1, max=float('inf')):
    if len(sen) > 0:
        tokens = nltk.tokenize.word_tokenize(sen)
        tokens = [token.lower() for token in tokens if token.isalnum()]
        if min <= len(tokens) <= max:
            return tokens
    return None


def index_list_of_sentences(sentences: [str], emb: KeyedVectors, dataset_name):
    return [index_sentence(sentence, emb) for sentence in tqdm(sentences, f"tokenizing {dataset_name} data set")]


def index_sentence(sentence, emb):
    return np.asarray([get_token_index(emb, word) for word in sentence], dtype="int32")


def get_token_index(emb, word):
    if word in emb.key_to_index:
        return emb.key_to_index[word]
    else:
        return emb.key_to_index["<unk>"]
