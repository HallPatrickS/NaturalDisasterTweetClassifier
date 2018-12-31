import os
from os import walk
import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# from . import params
import params


def load_disaster_tweets(data_path, seed):
    for root, _, files in walk(data_path):
        for f_name in files:
            with open(os.path.join(root, f_name)) as f:
                for row in csv.reader(f):
                    # yield (clean(row[9]), params.LABELS[root])
                    yield (row[9], params.LABELS[root])


def verify_labels(val_labels):
    num_classes = params.NUM_CLASSES
    unexpect = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpect):
        raise ValueError(
            "Unexpected labels in data".format(unexpected_labels=unexpected_labels)
        )
    pass


def ngram_vectorize(train_texts, train_labels):
    """Vectorizes texts as n-gram vectors.
    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.
    # Arguments
    train_texts: list, training text strings.
    train_labels: np.ndarray, training labels.
    val_texts: list, validation text strings.
    # Returns
    X_vect: vextorized training and validation data
    """
    vectorizer = TfidfVectorizer(
        encoding="utf-8",
        strip_accents="unicode",
        lowercase=True,
        analyzer=params.TOKEN_MODE,  # Split text into word tokens.
        max_features=params.TOP_K,
        ngram_range=params.NGRAM_RANGE,
        dtype=np.float64,
        min_df=params.MIN_DOCUMENT_FREQUENCY,
    )

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(params.TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype(np.float64)
    return x_train.toarray()
