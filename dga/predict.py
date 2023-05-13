import pickle

import click
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

from .constants import MODEL_DIR
from .constants import VOCAB
from .utils import extract_features
from .utils import words_to_ngram_codes_for_model

tf.compat.v1.disable_v2_behavior()


@click.group(name="predict_embeddings")
def main():
    pass


@main.command(name="binary")
@click.argument("domain", type=str)
@click.option("-n", required=True, type=int, help="The n in the n-grams to find")
def train_binary(domain: str, n: int) -> None:
    directory_to_load = MODEL_DIR / f"binary_{n}_grams"
    if not directory_to_load.exists():
        raise ValueError(f"{directory_to_load} does not exist, could not find model")
    model: tf.keras.models.Sequential = tf.keras.models.load_model(directory_to_load / "model.h5")
    length = model.input_shape[1]
    X = words_to_ngram_codes_for_model([domain], VOCAB, n, length)
    y = model.predict(X, verbose=0)
    predictions = np.argmax(y, axis=1)
    prediction = predictions[0]
    if prediction == 0:
        print(f"{domain} is non-DGA")
    else:
        print(f"{domain} is DGA")


@main.command(name="multiclass")
@click.argument("domain", type=str)
@click.option("-n", required=True, type=int, help="The n in the n-grams to find")
def train_binary(domain: str, n: int) -> None:
    directory_to_load = MODEL_DIR / f"multiclass_{n}_grams"
    if not directory_to_load.exists():
        raise ValueError(f"{directory_to_load} does not exist, could not find model")
    model: tf.keras.models.Sequential = tf.keras.models.load_model(directory_to_load / "model.h5")
    with (directory_to_load / "encoder.p").open("rb") as stream:
        encoder: OneHotEncoder = pickle.load(stream)
    length = model.input_shape[1]
    X = words_to_ngram_codes_for_model([domain], VOCAB, n, length)
    y = model.predict(X, verbose=0)
    predictions = np.argmax(y, axis=1)
    prediction = predictions[0]
    print(f"{domain} is {encoder.categories_[prediction]}")


@main.command(name="features")
@click.argument("domain", type=str)
def predict_features(domain: str) -> None:
    directory_to_load = MODEL_DIR / f"binary_features_deep"
    if not directory_to_load.exists():
        raise ValueError(f"{directory_to_load} does not exist, could not find model")
    model: tf.keras.models.Sequential = tf.keras.models.load_model(directory_to_load / "model.h5")
    X = extract_features(pd.Series([domain])).to_numpy()
    X = X[:, :-1].astype(np.int32)
    y = model.predict(X)
    predictions = np.argmax(y, axis=1)
    prediction = predictions[0]
    if prediction == 0:
        print(f"{domain} is non-DGA")
    else:
        print(f"{domain} is DGA")


if __name__ == "__main__":
    main()
