import sys

import numpy as np
import tensorflow as tf
import tqdm
from keras import Model
from keras.callbacks import BackupAndRestore, TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, r2_score, mean_squared_error, mean_absolute_error, accuracy_score, \
    confusion_matrix
from tensorflow.python.framework.errors_impl import NotFoundError

from util import embeddings, config
from util.embeddings import GLOVE, GN_GLOVE, GP_GN_GLOVE, GP_GLOVE
from tasks.conll2003_data import get_conll2003_dataset, get_conll2003_test
from tasks.hate_speech_detection.hsd_data import get_hsd_dataset
from tasks.hate_speech_detection.hsd_model import get_hsd_model
from tasks.language_modeling import language_model
from tasks.language_modeling import lm_data
from tasks.named_entity_recognition.ner_model import get_ner_model
from tasks.parts_of_speech_tagging.pos_model import get_pos_model
from tasks.sentiment_analysis.sa_data import get_imdb_dataset_tokenized
from tasks.sentiment_analysis.sa_model import get_sa_model

FORCE_CPU_USAGE = False
GPU_AVAILABLE = tf.test.is_gpu_available()


def load_or_train_model_weights(model: Model,
                                train,
                                validation,
                                epochs,
                                directory,
                                model_name,
                                get_best_model=True,
                                early_stopping=False):
    model_path = f"{directory}/model/model"
    best_model_path = f"{directory}/best_model/model"
    log_dir = f"{directory}/logs"
    backup_dir = f"{directory}/backup/"
    try:
        print(f"trying to load {model_name}-model")
        model.load_weights(model_path)
        print(f"weights successfully loaded")
    except (ImportError, ValueError, NotFoundError):
        print(f"weights not found - training {model_name} model")
        backup_callback = BackupAndRestore(backup_dir)
        best_model_callback = ModelCheckpoint(best_model_path, save_weights_only=True, save_best_only=True)
        tensorboard_callback = TensorBoard(log_dir)
        callbacks = [backup_callback, best_model_callback, tensorboard_callback]
        if early_stopping:
            callbacks.append(EarlyStopping(patience=10))
        fit_with_correct_device(lambda: model.fit(train, validation_data=validation, epochs=epochs,
                                                  callbacks=callbacks))
        model.save_weights(model_path)
    if get_best_model:
        model.load_weights(best_model_path)
    return model


def fit_with_correct_device(fit_fn):
    if GPU_AVAILABLE and not FORCE_CPU_USAGE:
        print("USING GPU")
        with tf.device('/gpu:0'):
            fit_fn()
    elif FORCE_CPU_USAGE:
        print("USING CPU")
        with tf.device('/cpu:0'):
            fit_fn()
    else:
        print("USING DEFAULT DEVICE")
        fit_fn()


def do_lm_evaluation(emb, base_path, embedding_name):
    train, valid, test, all_tokens = lm_data.get_wiki_text_103_dataset_tokenized(emb)
    model = language_model.get_language_model(emb, all_tokens)
    model = load_or_train_model_weights(model=model,
                                        train=train,
                                        validation=valid,
                                        epochs=30,
                                        directory=f"{base_path}models/lm/{embedding_name}",
                                        model_name="language",
                                        get_best_model=False)

    if GPU_AVAILABLE and not FORCE_CPU_USAGE:
        print("testing LM on GPU")
        with tf.device('/gpu:0'):
            result = model.evaluate(test)
    else:
        print("testing LM on default device")
        result = model.evaluate(test)
    cross_entropy = result[0]
    print(f"crossentropy all batches {round(cross_entropy, 4)} perplexity all batches {round(np.exp(cross_entropy))}")


def do_sa_evaluation(emb, base_path, embedding_name):
    train, test, valid = get_imdb_dataset_tokenized(emb)
    model = get_sa_model(emb)
    model = load_or_train_model_weights(model=model,
                                        train=train,
                                        validation=valid,
                                        epochs=100,
                                        directory=f"{base_path}models/sa/{embedding_name}",
                                        model_name="sentiment analysis",
                                        get_best_model=True,
                                        early_stopping=True)
    print("evaluating sa model")
    model.evaluate(test)
    y_true = []
    y_pred = []
    for X, Y in tqdm.tqdm(test, "evaluating SA model"):
        y_pred.extend(np.where(model.predict(X) >= 0.5, 1, 0).flatten().tolist())
        y_true.extend(Y)
    print(f"Sentiment Analysis {embedding_name} Accuracy: {accuracy_score(y_true, y_pred)}")
    print(
        f"Sentiment Analysis {embedding_name} Classification Report\n {classification_report(y_true, y_pred, digits=4)}")
    print(f"Sentiment Analysis {embedding_name} Confusion Matrix \n {confusion_matrix(y_true, y_pred)}")


def do_ner_evaluation(embedding, base_path, embedding_name):
    train, valid = get_conll2003_dataset(embedding, "ner")
    model = get_ner_model(embedding)
    model = load_or_train_model_weights(model=model,
                                        train=train,
                                        validation=valid,
                                        epochs=100,
                                        directory=f"{base_path}models/ner/{embedding_name}",
                                        model_name="named entity recognition",
                                        get_best_model=True,
                                        early_stopping=True)
    test_X, test_Y = get_conll2003_test(embedding, "ner")
    y_pred = []
    y_true = []
    for i, seq in tqdm.tqdm(enumerate(test_X), "evaluating NER model"):
        targets = test_Y[i]
        pred_y = np.argmax(model(seq.reshape(1, -1)), 2)
        y_pred.extend(pred_y[0].tolist())
        y_true.extend(targets.tolist())
    y_true_labels = [config.NER_TAGS[tag] for tag in y_true]
    y_pred_labels = [config.NER_TAGS[tag] for tag in y_pred]

    print(f" {classification_report(y_true_labels, y_pred_labels, digits=4, labels=list(config.NER_TAGS.values()))}")


def do_pos_evaluation(embedding, base_path, embedding_name):
    train, valid = get_conll2003_dataset(embedding, "pos")
    model = get_pos_model(embedding)
    model = load_or_train_model_weights(model=model,
                                        train=train,
                                        validation=valid,
                                        epochs=100,
                                        directory=f"{base_path}models/pos/{embedding_name}",
                                        model_name="parts-of-speech tagging",
                                        get_best_model=True,
                                        early_stopping=True)
    test_X, test_Y = get_conll2003_test(embedding, "pos")
    y_pred = []
    y_true = []
    for i, seq in tqdm.tqdm(enumerate(test_X), "evaluating POS model"):
        targets = test_Y[i]
        pred_y = np.argmax(model(seq.reshape(1, -1)), 2)
        y_pred.extend(pred_y[0].tolist())
        y_true.extend(targets.tolist())
    y_true_labels = [config.POS_TAGS[tag] for tag in y_true]
    y_pred_labels = [config.POS_TAGS[tag] for tag in y_pred]
    print(f"Named Entity Recognition {embedding_name} Accuracy {accuracy_score(y_true, y_pred)}")
    print(f"Named Entity Recognition {embedding_name} Classification Report: \n{classification_report(y_true_labels, y_pred_labels, digits=4, labels=list(config.POS_TAGS.values()))}")


def do_hsd_evaluation(embedding, base_path, embedding_name):
    train, test, valid = get_hsd_dataset(embedding)
    model = get_hsd_model(embedding)
    model = load_or_train_model_weights(model=model,
                                        train=train,
                                        validation=valid,
                                        epochs=100,
                                        directory=f"{base_path}models/hsd/{embedding_name}",
                                        model_name="hate speech detection",
                                        early_stopping=True)
    print("evaluating hate speech detection model")
    model.evaluate(test)
    y_true = []
    y_pred = []
    for X, Y in tqdm.tqdm(test, "evaluating HSD model"):
        y_pred.extend(model.predict(X))
        y_true.extend(Y)
    print(f"Hate Speech Detection {embedding_name} R_SQUARE: {r2_score(y_true, y_pred)}")
    print(f"Hate Speech Detection {embedding_name} MSE: {mean_squared_error(y_true, y_pred)}")
    print(f"Hate Speech Detection {embedding_name} MAE: {mean_absolute_error(y_true, y_pred)}")


def main():
    embedding_name = sys.argv[1]
    model_type = sys.argv[2]
    base_path = sys.argv[3]

    if embedding_name not in [GLOVE, GN_GLOVE, GP_GN_GLOVE, GP_GLOVE]:
        raise ValueError(f"please choose one of {[GLOVE, GN_GLOVE, GP_GN_GLOVE, GP_GLOVE]} as embedding name")

    embedding = embeddings.get_embedding_model(base_path, embedding_name)

    if model_type == "lm":
        do_lm_evaluation(embedding, base_path, embedding_name)
    elif model_type == "sa":
        do_sa_evaluation(embedding, base_path, embedding_name)
    elif model_type == "ner":
        do_ner_evaluation(embedding, base_path, embedding_name)
    elif model_type == "hsd":
        do_hsd_evaluation(embedding, base_path, embedding_name)
    elif model_type == "pos":
        do_pos_evaluation(embedding, base_path, embedding_name)
    else:
        print("invalid model type, please choose from: (sa, lm, ner, hsd, pos)")


if __name__ == '__main__':
    main()
