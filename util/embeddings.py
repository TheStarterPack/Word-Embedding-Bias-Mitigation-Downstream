import numpy as np
from gensim.models import KeyedVectors

from util.config import PADDING_WORD

GLOVE_EMBEDDING_PATH = "embeddings/glove.txt"
GN_GLOVE_EMBEDDING_PATH = "embeddings/gn-glove.txt"
GP_GLOVE_EMBEDDING_PATH = "embeddings/gp-glove.txt"
GP_GN_GLOVE_EMBEDDING_PATH = "embeddings/gp-gn-glove.txt"

GLOVE = "glove"
GN_GLOVE = "gn-glove"
GP_GLOVE = "gp-glove"
GP_GN_GLOVE = "gp-gn-glove"

NAME_TO_PATH = {
    GLOVE: GLOVE_EMBEDDING_PATH,
    GN_GLOVE: GN_GLOVE_EMBEDDING_PATH,
    GP_GLOVE: GP_GLOVE_EMBEDDING_PATH,
    GP_GN_GLOVE: GP_GN_GLOVE_EMBEDDING_PATH
}


def get_embedding_model(base_path, embedding_name):
    print(f"reading {embedding_name} embeddings from file")
    path = base_path + NAME_TO_PATH[embedding_name]
    embedding = KeyedVectors.load_word2vec_format(path)
    embedding.add_vector(PADDING_WORD, np.zeros(embedding.vectors.shape[1]))
    return embedding
