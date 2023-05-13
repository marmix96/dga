import pickle

import click
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .constants import BINARY_EXPLAINABILITY_SAMPLES
from .constants import DATA
from .constants import MODEL_DIR
from .constants import MULTICLASS_EXPLAINABILITY_SAMPLES
from .constants import TRAIN_EXPLAINABILITY_SAMPLES
from .constants import VOCAB
from .utils import extract_features
from .utils import ngram_codes_to_words
from .utils import split_model
from .utils import words_to_ngram_codes_for_model
from .utils import words_to_ngrams

tf.compat.v1.disable_v2_behavior()


def preprocess_data() -> pd.DataFrame:
    """
    Preprocess the data by reading CSV files, adding columns, removing duplicates,
    and filtering domains based on length.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    dfs = []
    for file in DATA[0]:
        df = pd.read_csv(file, header=None, names=["domain"], usecols=[1])
        df["algorithm"] = None
        df["label"] = 0
        dfs.append(df)

    for file in DATA[1]:
        df = pd.read_csv(file, header=0)
        df["label"] = 1
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df.drop_duplicates(subset="domain", keep="first")
    df = df[df["domain"].str.len() > 3]
    df = df.reset_index(drop=True)

    return df


def train_save_binary_n_grams(df: pd.DataFrame, n: int, length: int, vocab: list[str]):
    """
    Train a binary n-gram model using the provided DataFrame and save the trained model, data, and encoder.

    The function performs the following steps:
    1. Filters the DataFrame to keep domains with lengths less than or equal to the specified length.
    2. Extracts the domain names and labels from the DataFrame.
    3. Saves the DataFrame as a CSV file.
    4. Converts the domain names into binary n-gram codes using the specified vocabulary and fixed length.
    5. One-hot encodes the labels.
    6. Splits the data into training and testing sets.
    7. Saves the training and testing data as compressed NumPy arrays.
    8. Saves the encoder used for label encoding.
    9. Creates a sequential model with embedding, LSTM, and dense layers.
    10. Compiles the model with suitable loss, optimizer, and metrics.
    11. Trains the model using the training data, with validation on the testing data.
    12. Saves the trained model with the best performance on the validation set.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        n (int): The length of n-grams.
        length (int): The fixed length for padding.
        vocab (list[str]): The vocabulary list.
    """
    directory_to_save = MODEL_DIR / f"binary_{n}_grams"
    directory_to_save.mkdir(parents=True, exist_ok=True)

    # Find data to use
    print("Find data to use")
    df = df[df["domain"].str.len() <= length]
    domains = df["domain"].to_list()
    y = df["label"].to_numpy()

    # Save DF
    print("Save dataframe")
    df.to_csv(directory_to_save / "df.csv", index=False)

    # Create X and y data
    print("Create X and y")
    X = words_to_ngram_codes_for_model(domains, vocab, n, length)
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32)
    encoder.fit(y.reshape(-1, 1))
    y = encoder.transform(y.reshape(-1, 1))

    # Split data for training and testing
    print("Split train test data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    print("Save train and test data")
    np.savez_compressed(directory_to_save / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(directory_to_save / "test.npz", X=X_test, y=y_test)

    print("Save encoder")
    with (directory_to_save / "encoder.p").open("wb") as stream:
        pickle.dump(encoder, stream)

    # Create model
    print("Create model")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[length]))
    model.add(tf.keras.layers.Embedding(input_dim=(len(vocab) ** n) + 1, output_dim=4))
    model.add(tf.keras.layers.Reshape([1, -1]))
    model.add(tf.keras.layers.LSTM(units=128, activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=32, activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=2, activation="softmax"))
    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),
        optimizer="adam",
        metrics=[tf.metrics.CategoricalAccuracy()],
    )
    model.summary()

    # Train model
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=directory_to_save / "model.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_freq="epoch",
        save_weights_only=False,
    )
    model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        batch_size=128,
        epochs=3,
        callbacks=[checkpoint_callback],
    )


def predict_save_binary_n_grams(n: int):
    """
    Predict labels for test data using a trained binary n-gram model and save the predictions.

    The function performs the following steps:
    1. Loads the trained model and test data.
    2. Performs predictions on the test data using the model.
    3. Saves the predicted labels as compressed NumPy array.

    Args:
        n (int): The length of n-grams used in the binary n-gram model.
    """
    directory_to_save = MODEL_DIR / f"binary_{n}_grams"

    print("Load model and test data")
    model: tf.keras.Sequential = tf.keras.models.load_model(directory_to_save / "model.h5")
    test_data = np.load(directory_to_save / "test.npz", allow_pickle=True)
    X_test = test_data["X"]

    print("Predict test data")
    y_pred = model.predict(X_test, batch_size=128, verbose=2)

    print("Save predictions")
    np.savez_compressed(directory_to_save / "predictions.npz", arr=y_pred)


def explain_save_binary_n_grams(n: int, length: int, vocab: list[str]):
    """
    Explain and save the binary n-gram model by generating SHAP values and saving the explanations.

    The function performs the following steps:
    1. Loads the trained model, training data, test data, and predicted labels.
    2. Splits the model into an embedding model and an explainable model.
    3. Prepares data for explanation by selecting a subset of training and test data.
    4. Generates SHAP values using the DeepExplainer.
    5. Creates a SHAP DataFrame with averaged values for each n-gram.
    6. Saves the explainer, SHAP values, and SHAP DataFrame.

    Args:
        n (int): The length of n-grams used in the binary n-gram model.
        length (int): The fixed length used for padding.
        vocab (list[str]): The vocabulary list.
    """
    directory_to_save = MODEL_DIR / f"binary_{n}_grams"

    print("Load model, train and test data")
    model: tf.keras.Sequential = tf.keras.models.load_model(directory_to_save / "model.h5")
    train_data = np.load(directory_to_save / "train.npz", allow_pickle=True)
    X_train = train_data["X"]
    test_data = np.load(directory_to_save / "test.npz", allow_pickle=True)
    X_test, y_test = test_data["X"], test_data["y"]
    pred_data = np.load(directory_to_save / "predictions.npz", allow_pickle=True)
    y_pred = pred_data["arr"]

    # Split model
    print("Split model")
    embedding_model, explainable_model = split_model(model)

    # Explain model
    print("Prepare data for explanation")
    X_train_explainable = X_train[:TRAIN_EXPLAINABILITY_SAMPLES]
    X_train_explainable_embeddings = embedding_model.predict(X_train_explainable)
    X_test_explainable = X_test[:BINARY_EXPLAINABILITY_SAMPLES]
    y_test_explainable = y_test[:BINARY_EXPLAINABILITY_SAMPLES]
    y_pred_explainable = y_pred[:BINARY_EXPLAINABILITY_SAMPLES]
    X_test_explainable_embeddings = embedding_model.predict(X_test_explainable)
    X_test_explainable_domains = ngram_codes_to_words(X_test_explainable, vocab, n)
    X_test_explainable_ngrams = words_to_ngrams(X_test_explainable_domains, n)

    print("Create explainer and generate shap values")
    explainer = shap.DeepExplainer(
        (explainable_model.inputs[0], explainable_model.outputs[0]),
        X_train_explainable_embeddings,
    )
    shap_values = explainer.shap_values(X_test_explainable_embeddings)

    # Create shap DF
    print(f"Find {n}-gram values for each train datum")
    domain_series = []
    for ngrams, shap_value in zip(X_test_explainable_ngrams, shap_values[1]):
        shap_value = np.mean(shap_value, axis=1)
        df_values = []
        for ngram, shapley in zip(ngrams, shap_value):
            df_values.append({"ngram": ngram, "shapley": shapley})
        domain_df = pd.DataFrame(df_values)
        domain_df = domain_df.groupby(by=["ngram"]).mean().reset_index()
        domain_df = domain_df.pivot_table(columns=["ngram"], values=["shapley"]).reset_index(drop=True)
        domain_series.append(domain_df.iloc[0])

    print("Create shap dataframe")
    shap_values_df = pd.DataFrame(domain_series).reset_index(drop=True).sort_index(axis=1)
    shap_values_df.index = pd.Index(X_test_explainable_domains, name="domain")

    print("Save explainer, shap values and shap df")
    np.savez_compressed(
        directory_to_save / "explainable.npz",
        X=X_test_explainable,
        y_true=y_test_explainable,
        y_pred=y_pred_explainable,
    )
    np.savez_compressed(directory_to_save / "expected_value.npz", arr=explainer.expected_value)
    np.savez_compressed(directory_to_save / "shap_values.npz", arr=shap_values)
    shap_values_df.to_csv(directory_to_save / "shap_values_df.csv")


def train_explain_save_binary_n_grams(df: pd.DataFrame, n: int, length: int, vocab: list[str]):
    train_save_binary_n_grams(df, n, length, vocab)
    predict_save_binary_n_grams(n)
    explain_save_binary_n_grams(n, length, vocab)


def train_save_multiclass_n_grams(df: pd.DataFrame, n: int, length: int, vocab: list[str]):
    """
    Train and save a multiclass n-gram model using the provided DataFrame and save the trained model, data, and encoder.

    The function performs the following steps:
    1. Filters the DataFrame to remove rows with missing algorithm values.
    2. Selects the algorithms that have at least 10,000 instances.
    3. Filters the DataFrame to keep domains with lengths less than or equal to the specified length and the selected algorithms.
    4. Extracts the domain names and algorithm labels from the DataFrame.
    5. Saves the DataFrame as a CSV file.
    6. Converts the domain names into n-gram codes using the specified vocabulary and fixed length.
    7. One-hot encodes the algorithm labels.
    8. Splits the data into training and testing sets.
    9. Saves the training and testing data as compressed NumPy arrays.
    10. Saves the encoder used for label encoding.
    11. Creates a sequential model with embedding, LSTM, and dense layers for multiclass classification.
    12. Compiles the model with suitable loss, optimizer, and metrics.
    13. Trains the model using the training data, with validation on the testing data.
    14. Saves the trained model with the best performance on the validation set.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        n (int): The length of n-grams.
        length (int): The fixed length for padding.
        vocab (list[str]): The vocabulary list.
    """
    directory_to_save = MODEL_DIR / f"multiclass_{n}_grams"
    directory_to_save.mkdir(parents=True, exist_ok=True)

    # Find data to use
    print("Find data to use")
    df = df.dropna(axis=0, subset="algorithm")
    algorithms = df["algorithm"].value_counts().loc[lambda count: count >= 1e4].index.tolist()
    df = df[df["algorithm"].isin(algorithms)]
    df = df[df["domain"].str.len() <= length]
    domains = df["domain"].to_list()
    y = df["algorithm"].to_numpy()

    # Save DF
    print("Save DF")
    df.to_csv(directory_to_save / "df.csv", index=False)

    # Create X and y data
    print("Create X and y")
    X = words_to_ngram_codes_for_model(domains, vocab, n, length)
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32)
    encoder.fit(y.reshape(-1, 1))
    y = encoder.transform(y.reshape(-1, 1))
    output_length = y.shape[1]

    # Split data for training and testing
    print("Split train test data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    print("Save train and test data")
    np.savez_compressed(directory_to_save / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(directory_to_save / "test.npz", X=X_test, y=y_test)

    print("Save encoder")
    with (directory_to_save / "encoder.p").open("wb") as stream:
        pickle.dump(encoder, stream)

    # Create model
    print("Create model")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[length]))
    model.add(tf.keras.layers.Embedding(input_dim=(len(vocab) ** n) + 1, output_dim=4))
    model.add(tf.keras.layers.Reshape([1, -1]))
    model.add(tf.keras.layers.LSTM(units=128, activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=32, activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=output_length, activation="softmax"))
    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),
        optimizer="adam",
        metrics=[tf.metrics.CategoricalAccuracy()],
    )
    model.summary()

    # Train model
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=directory_to_save / "model.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_freq="epoch",
        save_weights_only=False,
    )
    model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        batch_size=128,
        epochs=3,
        callbacks=[checkpoint_callback],
    )


def predict_save_multiclass_n_grams(n: int):
    """
    Predict labels for test data using a trained multiclass n-gram model and save the predictions.

    The function performs the following steps:
    1. Loads the trained model and test data.
    2. Performs predictions on the test data using the model.
    3. Saves the predicted labels as compressed NumPy array.

    Args:
        n (int): The length of n-grams used in the multiclass n-gram model.
    """
    directory_to_save = MODEL_DIR / f"multiclass_{n}_grams"

    print("Load model and test data")
    model: tf.keras.Sequential = tf.keras.models.load_model(directory_to_save / "model.h5")
    test_data = np.load(directory_to_save / "test.npz", allow_pickle=True)
    X_test = test_data["X"]

    print("Predict test data")
    y_pred = model.predict(X_test, batch_size=128, verbose=2)

    print("Save predictions")
    np.savez_compressed(directory_to_save / "predictions.npz", arr=y_pred)


def explain_save_algorithm_n_grams(
    n: int,
    length: int,
    vocab: list[str],
    algorithm: str,
    category: int,
    explainer: shap.DeepExplainer,
):
    """
    Explain and save the n-gram model for a specific algorithm by generating SHAP values and saving the explanations.

    The function performs the following steps:
    1. Loads the necessary data, including the DataFrame, model, and test data.
    2. Splits the model into an embedding model and an explainable model.
    3. Prepares the data for explanation by selecting a subset of test data related to the specified algorithm.
    4. Generates SHAP values using the DeepExplainer for the specified category.
    5. Creates a SHAP DataFrame with averaged values for each n-gram.
    6. Saves the explainer, SHAP values, and SHAP DataFrame.

    Args:
        n (int): The length of n-grams used in the n-gram model.
        length (int): The fixed length used for padding.
        vocab (list[str]): The vocabulary list.
        algorithm (str): The algorithm to explain.
        category (int): The category to explain (corresponding to the output neuron).
        explainer (shap.DeepExplainer): The explainer object used for generating SHAP values.
    """
    print(f"Explain {algorithm}")
    directory_to_save = MODEL_DIR / f"multiclass_{n}_grams"

    print("Load dataframe, model, train and test data")
    df = pd.read_csv(directory_to_save / "df.csv", header=0, index_col=None)
    model: tf.keras.Sequential = tf.keras.models.load_model(directory_to_save / "model.h5")
    test_data = np.load(directory_to_save / "test.npz", allow_pickle=True)
    X_test, y_test = test_data["X"], test_data["y"]
    pred_data = np.load(directory_to_save / "predictions.npz", allow_pickle=True)
    y_pred = pred_data["arr"]

    # Split model
    print("Split model")
    embedding_model, _ = split_model(model)

    # Explain model
    print("Prepare data for explanation")
    X_test_domains = ngram_codes_to_words(X_test, vocab, n)
    X_test_algo_df = df[(df["domain"].isin(X_test_domains)) & (df["algorithm"] == algorithm)]
    X_test_domains_algo_from_df = set(X_test_algo_df["domain"].to_numpy())
    all_test_data = [
        (domain, y, pred)
        for domain, y, pred in zip(X_test_domains, y_test, y_pred)
        if domain in X_test_domains_algo_from_df
    ]
    all_test_data_explainable = all_test_data[:MULTICLASS_EXPLAINABILITY_SAMPLES]
    X_test_explainable_domains, y_test_explainable, y_pred_explainable = zip(*all_test_data_explainable)
    X_test_explainable_domains = np.asarray(X_test_explainable_domains)
    y_test_explainable = np.asarray(y_test_explainable)
    y_pred_explainable = np.asarray(y_pred_explainable)
    X_test_explainable = words_to_ngram_codes_for_model(X_test_explainable_domains, vocab, n, length)
    X_test_explainable_ngrams = words_to_ngrams(X_test_explainable_domains, n)
    X_test_explainable_embeddings = embedding_model.predict(X_test_explainable)

    shap_values = explainer.shap_values(X_test_explainable_embeddings)

    # Create shap DF
    print(f"Find {n}-gram values for each train datum")
    domain_series = []
    for ngrams, shap_value in zip(X_test_explainable_ngrams, shap_values[category]):
        shap_value = np.mean(shap_value, axis=1)
        df_values = []
        for ngram, shapley in zip(ngrams, shap_value):
            df_values.append({"ngram": ngram, "shapley": shapley})
        domain_df = pd.DataFrame(df_values)
        domain_df = domain_df.groupby(by=["ngram"]).mean().reset_index()
        domain_df = domain_df.pivot_table(columns=["ngram"], values=["shapley"]).reset_index(drop=True)
        domain_series.append(domain_df.iloc[0])

    print("Create shap dataframe")
    shap_values_df = pd.DataFrame(domain_series).reset_index(drop=True).sort_index(axis=1)
    shap_values_df.index = pd.Index(X_test_explainable_domains, name="domain")

    print("Save explainer, shap values and shap df")
    (directory_to_save / algorithm).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory_to_save / algorithm / "explainable.npz",
        X=X_test_explainable,
        y_true=y_test_explainable,
        y_pred=y_pred_explainable,
    )
    np.savez_compressed(directory_to_save / algorithm / "expected_value.npz", arr=explainer.expected_value)
    np.savez_compressed(directory_to_save / algorithm / "shap_values.npz", arr=shap_values)
    shap_values_df.to_csv(directory_to_save / algorithm / "shap_values_df.csv")


def explain_save_multiclass_n_grams(n: int, length: int, vocab: list[str]):
    """
    Explain and save the multiclass n-gram model for each algorithm by generating SHAP values and saving the explanations.

    The function performs the following steps:
    1. Loads the encoder used for one-hot encoding the algorithm labels.
    2. Loads the trained model and training data.
    3. Splits the model into an embedding model and an explainable model.
    4. Prepares the training data for explanation by selecting a subset for explanation.
    5. Creates the explainer using the explainable model and the training data.
    6. Iterates over each algorithm category and invokes the `explain_save_algorithm_n_grams` function to explain
       and save the n-gram model for that algorithm.

    Args:
        n (int): The length of n-grams used in the multiclass n-gram model.
        length (int): The fixed length used for padding.
        vocab (list[str]): The vocabulary list.
    """
    directory_to_save = MODEL_DIR / f"multiclass_{n}_grams"

    print("Load encoder")
    with (directory_to_save / "encoder.p").open("rb") as stream:
        encoder: OneHotEncoder = pickle.load(stream)

    print("Load model and train data")
    model: tf.keras.Sequential = tf.keras.models.load_model(directory_to_save / "model.h5")
    train_data = np.load(directory_to_save / "train.npz", allow_pickle=True)
    X_train = train_data["X"]
    X_train_explainable = X_train[:TRAIN_EXPLAINABILITY_SAMPLES]

    # Split model
    print("Split model")
    embedding_model, explainable_model = split_model(model)

    print("Prepare train data for explanation")
    X_train_explainable = X_train[:TRAIN_EXPLAINABILITY_SAMPLES]
    X_train_explainable_embeddings = embedding_model.predict(X_train_explainable)

    print("Create explainer and generate shap values")
    explainer = shap.DeepExplainer(
        (explainable_model.inputs[0], explainable_model.outputs[0]),
        X_train_explainable_embeddings,
    )

    for i, algorithm in enumerate(encoder.categories_[0]):
        explain_save_algorithm_n_grams(n, length, vocab, algorithm, i, explainer)


def train_explain_save_multiclass_n_grams(df: pd.DataFrame, n: int, length: int, vocab: list[str]):
    train_save_multiclass_n_grams(df, n, length, vocab)
    predict_save_multiclass_n_grams(n)
    explain_save_multiclass_n_grams(n, length, vocab)


def train_save_binary_features_deep(df: pd.DataFrame):
    """
    Train and save a deep binary features model using the provided DataFrame.

    The function performs the following steps:
    1. Extracts features from the domain column of the DataFrame using the `extract_features` function.
    2. Creates the input data X and the target labels y.
    3. Saves the DataFrame and the encoder used for one-hot encoding the labels.
    4. Splits the data into training and testing sets.
    5. Creates a deep learning model.
    6. Trains the model on the training data and evaluates it on the testing data.
    7. Saves the trained model.

    Args:
        df (pd.DataFrame): The DataFrame containing the domain and label columns.
    """
    directory_to_save = MODEL_DIR / f"binary_features_deep"
    directory_to_save.mkdir(parents=True, exist_ok=True)

    print("Extract features")
    if (directory_to_save / "features.csv").exists():
        features_df = pd.read_csv(directory_to_save / "features.csv", header=0, index_col=None)
    else:
        features_df = extract_features(df["domain"])

    # Create X and y
    print("Create X and y")
    X = features_df.to_numpy()
    y = df["label"].to_numpy()
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32)
    encoder.fit(y.reshape(-1, 1))
    y = encoder.transform(y.reshape(-1, 1))

    # One of the columns will be the domain which will be excluded from the input data
    length = X.shape[1] - 1

    # Save DF
    print("Save DF")
    df.to_csv(directory_to_save / "df.csv", index=False)
    features_df.to_csv(directory_to_save / "features.csv", index=False)

    print("Save encoder")
    with (directory_to_save / "encoder.p").open("wb") as stream:
        pickle.dump(encoder, stream)

    # Split data for training and testing
    print("Split train test data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    print("Save train test data")
    np.savez_compressed(directory_to_save / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(directory_to_save / "test.npz", X=X_test, y=y_test)

    # Create model
    print("Create model")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[length]))
    model.add(tf.keras.layers.Dense(units=128, activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=32, activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=2, activation="softmax"))
    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),
        optimizer="rmsprop",
        metrics=[tf.metrics.CategoricalAccuracy()],
    )
    model.summary()

    # Train model
    print("Train model")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=directory_to_save / "model.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_freq="epoch",
        save_weights_only=False,
    )
    model.fit(
        x=X_train[:, :-1].astype(np.int32),
        y=y_train,
        validation_data=(X_test[:, :-1].astype(np.int32), y_test),
        batch_size=128,
        epochs=3,
        callbacks=[checkpoint_callback],
    )


def predict_save_binary_features_deep():
    """
    Predict and save the binary features deep model on the test data.

    The function performs the following steps:
    1. Loads the trained model and the test data.
    2. Performs predictions on the test data.
    3. Saves the predictions.
    """
    directory_to_save = MODEL_DIR / f"binary_features_deep"

    print("Load model and test data")
    model: tf.keras.Sequential = tf.keras.models.load_model(directory_to_save / "model.h5")
    test_data = np.load(directory_to_save / "test.npz", allow_pickle=True)
    X_test = test_data["X"]

    print("Predict test data")
    y_pred = model.predict(X_test[:, :-1].astype(np.int32), batch_size=128, verbose=2)

    print("Save predictions")
    np.savez_compressed(directory_to_save / "predictions.npz", arr=y_pred)


def explain_save_binary_features_deep():
    """
    Explain and save the binary features deep model.

    The function performs the following steps:
    1. Loads the dataframe, trained model, and train/test data.
    2. Prepares the data for explanation.
    3. Creates an explainer and generates SHAP values.
    4. Creates a SHAP DataFrame.
    5. Saves the explainer, SHAP values, and SHAP DataFrame.
    """
    directory_to_save = MODEL_DIR / f"binary_features_deep"

    print("Load dataframe, model, train and test data")
    features_df = pd.read_csv(directory_to_save / "features.csv")
    model: tf.keras.Sequential = tf.keras.models.load_model(directory_to_save / "model.h5")
    train_data = np.load(directory_to_save / "train.npz", allow_pickle=True)
    X_train = train_data["X"]
    test_data = np.load(directory_to_save / "test.npz", allow_pickle=True)
    X_test, y_test = test_data["X"], test_data["y"]
    pred_data = np.load(directory_to_save / "predictions.npz", allow_pickle=True)
    y_pred = pred_data["arr"]

    print("Prepare data for explanation")
    X_train_explainable = X_train[:TRAIN_EXPLAINABILITY_SAMPLES]
    X_test_explainable = X_test[:BINARY_EXPLAINABILITY_SAMPLES]
    y_test_explainable = y_test[:BINARY_EXPLAINABILITY_SAMPLES]
    y_pred_explainable = y_pred[:BINARY_EXPLAINABILITY_SAMPLES]

    print("Create explainer and generate shap values")
    explainer = shap.DeepExplainer(
        (model.inputs[0], model.outputs[0]),
        X_train_explainable[:, :-1].astype(np.int32),
    )
    shap_values = explainer.shap_values(X_test_explainable[:, :-1].astype(np.int32))

    # Create shap DF
    shap_values_df = pd.DataFrame(shap_values[1], columns=features_df.columns.values[:-1])

    print("Save explainer, shap values and shap df")
    np.savez_compressed(
        directory_to_save / "explainable.npz",
        X=X_test_explainable,
        y_true=y_test_explainable,
        y_pred=y_pred_explainable,
    )
    np.savez_compressed(directory_to_save / "expected_value.npz", arr=explainer.expected_value)
    np.savez_compressed(directory_to_save / "shap_values.npz", arr=shap_values)
    shap_values_df.to_csv(directory_to_save / "shap_values_df.csv")


def train_explain_save_binary_features_deep(df: pd.DataFrame):
    train_save_binary_features_deep(df)
    predict_save_binary_features_deep()
    explain_save_binary_features_deep()


@click.group(name="train_embeddings")
def main():
    pass


@main.command(name="binary")
@click.option("-n", required=True, multiple=True, type=int, help="The n in the n-grams to find")
@click.option("-l", "--length", type=int, default=256, help="The length of the embeddings")
def train_binary(n: list[int], length: int) -> None:
    df = preprocess_data()
    for s_n in n:
        train_explain_save_binary_n_grams(df, s_n, length, VOCAB)


@main.command(name="multiclass")
@click.option("-n", required=True, multiple=True, type=int, help="The n in the n-grams to find")
@click.option("-l", "--length", type=int, default=256, help="The length of the embeddings")
def train_multiclass(n: list[int], length: int) -> None:
    df = preprocess_data()
    for s_n in n:
        train_explain_save_multiclass_n_grams(df, s_n, length, VOCAB)


@main.command(name="features")
def train_features():
    df = preprocess_data()
    train_explain_save_binary_features_deep(df)


if __name__ == "__main__":
    main()
