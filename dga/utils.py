import itertools
import re
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from .constants import CONSONANTS
from .constants import DIGITS
from .constants import HYPHENS
from .constants import VOWELS


def count_chars(s: str, valid: list[str]) -> int:
    """
    Count the number of characters in a string that are present in a given list of valid characters.

    Args:
        s (str): The input string to count characters from.
        valid (list[str]): The list of valid characters.

    Returns:
        int: The count of characters in the string that are present in the valid characters list.
    """
    return len([l for l in s if l in valid])


def count_sequence(s: str, valid: list[str], agg: Callable[..., int]) -> int:
    """
    Count the length of sequences in a string that consist of characters from a given list of valid characters,
    and apply an aggregation function to the lengths.

    Args:
        s (str): The input string to count sequences from.
        valid (list[str]): The list of valid characters.
        agg (Callable[..., int]): The aggregation function to apply to the lengths of the sequences.

    Returns:
        int: The result of applying the aggregation function to the lengths of the sequences.
    """
    regex = re.compile(f"([{''.join(valid)}]+)")
    matches = regex.findall(s)
    if not matches:
        return 0
    return agg([len(match) for match in matches])


def extract_features(series: pd.Series) -> pd.DataFrame:
    """
    Extract various features from a pandas Series of strings and return them as a DataFrame.

    Args:
        series (pd.Series): The input series containing strings to extract features from.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted features.
    """
    return pd.DataFrame(
        {
            "vowels": series.apply(count_chars, valid=VOWELS),
            "consonants": series.apply(count_chars, valid=CONSONANTS),
            "hyphens": series.apply(count_chars, valid=HYPHENS),
            "digits": series.apply(count_chars, valid=DIGITS),
            "min_vowels": series.apply(count_sequence, valid=VOWELS, agg=min),
            "max_vowels": series.apply(count_sequence, valid=VOWELS, agg=max),
            "min_consonants": series.apply(count_sequence, valid=CONSONANTS, agg=min),
            "max_consonants": series.apply(count_sequence, valid=CONSONANTS, agg=max),
            "min_hyphens": series.apply(count_sequence, valid=HYPHENS, agg=min),
            "max_hyphens": series.apply(count_sequence, valid=HYPHENS, agg=max),
            "min_digits": series.apply(count_sequence, valid=DIGITS, agg=min),
            "max_digits": series.apply(count_sequence, valid=DIGITS, agg=max),
            "dns_labels": series.apply(lambda x: len(x.split("."))),
            "domain": series.apply(lambda x: x),
        }
    )


def create_ngram_codes(vocab: list[str], n: int) -> dict[str, int]:
    """
    Create n-gram codes for a given vocabulary list.

    Args:
        vocab (list[str]): The vocabulary list.
        n (int): The length of n-grams.

    Returns:
        dict[str, int]: A dictionary mapping n-grams to their corresponding codes.
    """
    return {x: i for i, x in enumerate(["".join(p) for p in itertools.product(vocab, repeat=n)], start=1)}


def word_to_ngrams(word: str, n: int) -> list[str]:
    """
    Convert a word into a list of n-grams.

    Args:
        word (str): The input word.
        n (int): The length of n-grams.

    Returns:
        list[str]: A list of n-grams extracted from the word.
    """
    return [word[i : i + n] for i in range(len(word) - n + 1)]


def words_to_ngrams(words: list[str], n: int) -> list[list[str]]:
    """
    Convert a list of words into a list of n-grams for each word.

    Args:
        words (list[str]): The list of input words.
        n (int): The length of n-grams.

    Returns:
        list[list[str]]: A list of lists, where each inner list contains the n-grams for the corresponding word.
    """
    return [word_to_ngrams(word, n) for word in words]


def words_to_ngram_codes(words: list[str], vocab: list[str], n: int) -> list[list[int]]:
    """
    Convert a list of words into a list of n-gram codes for each word, based on a given vocabulary.

    Args:
        words (list[str]): The list of input words.
        vocab (list[str]): The vocabulary list.
        n (int): The length of n-grams.

    Returns:
        list[list[int]]: A list of lists, where each inner list contains the n-gram codes for the corresponding word.
    """
    ngram_codes = create_ngram_codes(vocab, n)
    words_ngrams = words_to_ngrams(words, n)
    words_ngrams = [[ngram_codes[ngram] for ngram in word_ngrams] for word_ngrams in words_ngrams]
    return words_ngrams


def words_to_ngram_codes_for_model(
    words: list[str], vocab: list[str], n: int, length: int
) -> np.ndarray[None, np.float64]:
    """
    Convert a list of words into a padded array of n-gram codes for a machine learning model,
    based on a given vocabulary and fixed length.

    Args:
        words (list[str]): The list of input words.
        vocab (list[str]): The vocabulary list.
        n (int): The length of n-grams.
        length (int): The fixed length for padding.

    Returns:
        np.ndarray: A numpy array of shape (num_samples, length) containing the padded n-gram codes,
            with each element as np.float64.
    """
    words_ngrams = words_to_ngram_codes(words, vocab, n)
    return tf.keras.utils.pad_sequences(words_ngrams, maxlen=length, padding="post", truncating="post").astype(
        np.float64
    )


def ngram_codes_to_ngrams(ngram_codes: list[list[float]], vocab: list[str], n: int) -> list[list[str]]:
    """
    Convert a list of n-gram codes to the corresponding n-grams, based on a given vocabulary.

    Args:
        ngram_codes (list[list[float]]): The list of n-gram codes.
        vocab (list[str]): The vocabulary list.
        n (int): The length of n-grams.

    Returns:
        list[list[str]]: A list of lists, where each inner list contains the n-grams for the corresponding n-gram codes.
    """
    codes_to_ngrams = {v: k for k, v in create_ngram_codes(vocab, n).items()}
    return [[codes_to_ngrams[int(ngram_code)] for ngram_code in tmp if int(ngram_code) > 0] for tmp in ngram_codes]


def ngrams_to_word(ngrams: list[str]) -> str:
    """
    Convert a list of n-grams to a word.

    Args:
        ngrams (list[str]): The list of n-grams.

    Returns:
        str: The word generated from the n-grams.
    """
    letters = [ngram[0] for ngram in ngrams]
    letters += ngrams[-1][1:]
    word = "".join(letters)
    return word


def ngram_codes_to_words(ngram_codes: list[list[float]], vocab: list[str], n: int) -> list[str]:
    """
    Convert a list of n-gram codes to words, based on a given vocabulary.

    Args:
        ngram_codes (list[list[float]]): The list of n-gram codes.
        vocab (list[str]): The vocabulary list.
        n (int): The length of n-grams.

    Returns:
        list[str]: A list of words generated from the n-gram codes.
    """
    return [ngrams_to_word(ngrams) for ngrams in ngram_codes_to_ngrams(ngram_codes, vocab, n)]


def split_model(model: tf.keras.Sequential) -> tuple[tf.keras.Sequential, tf.keras.Sequential]:
    """
    Split a sequential model into an embedding model and an explainable model.

    The embedding model consists of the first layer of the original model,
    while the explainable model consists of the remaining layers.

    Args:
        model (tf.keras.Sequential): The sequential model to be split.

    Returns:
        tuple[tf.keras.Sequential, tf.keras.Sequential]: A tuple containing the embedding model and the explainable model.
    """
    embedding_layer = model.layers[0]
    len = model.input_shape[1]
    embedding_model = tf.keras.Sequential()
    embedding_model.add(tf.keras.layers.Input(shape=[len]))
    embedding_model.add(embedding_layer)
    embedding_model.build()

    explainable_model = tf.keras.Sequential()
    explainable_model.add(tf.keras.layers.Input(shape=[len, 4]))
    for layer in model.layers[1:]:
        explainable_model.add(layer)
    explainable_model.build()

    return embedding_model, explainable_model
