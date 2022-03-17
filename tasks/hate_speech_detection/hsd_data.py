import tensorflow as tf
import tqdm
from datasets import load_dataset
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

from util.preprocessing import preprocess_sentence, index_list_of_sentences

LOG_TAG = "HSD_DATA"

def get_hsd_dataset(emb: KeyedVectors):
    print("Loading Measuring Hate Speech Dataset")
    train = load_dataset("ucberkeley-dlab/measuring-hate-speech", "ucberkeley-dlab--measuring-hate-speech")["train"]
    X = []
    y = []
    processed_comment_ids = set()
    for datapoint in tqdm.tqdm(train, "processing hsd data"):
        comment_id = datapoint["comment_id"]
        if comment_id in processed_comment_ids:
            continue
        sen = preprocess_sentence(datapoint["text"])
        if sen is not None:
            processed_comment_ids.add(comment_id)
            X.append(sen)
            y.append(datapoint["hate_speech_score"])

    X = index_list_of_sentences(X, emb, "measuring-hate-speech")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5923)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=7543)

    return tf.data.Dataset.from_tensor_slices((tf.ragged.constant(X_train), tf.ragged.constant(y_train))).shuffle(10000).batch(32), \
           tf.data.Dataset.from_tensor_slices((tf.ragged.constant(X_test), tf.ragged.constant(y_test))).shuffle(10000).batch(32), \
           tf.data.Dataset.from_tensor_slices((tf.ragged.constant(X_val), tf.ragged.constant(y_val))).shuffle(10000).batch(32)