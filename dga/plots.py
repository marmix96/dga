import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.preprocessing import OneHotEncoder

from .constants import VOCAB
from .utils import ngram_codes_to_words


def feature_plot_ngrams(
    shap_values_csv: Path,
    explainable_npz: Path,
    encoder_p: Path,
    absolute: bool = False,
) -> go.Figure:
    n = 20
    with encoder_p.open("rb") as stream:
        encoder: OneHotEncoder = pickle.load(stream)
    df = pd.read_csv(shap_values_csv, index_col=0)
    explainable_data = np.load(explainable_npz, allow_pickle=True)
    X = explainable_data["X"]
    y_true = encoder.inverse_transform(explainable_data["y_true"]).reshape([-1])
    y_pred = encoder.inverse_transform(explainable_data["y_pred"]).reshape([-1])
    domains = ngram_codes_to_words(X, VOCAB, len(df.columns.values[0]))
    data_df = pd.DataFrame({"domain": domains, "expected": y_true, "predicted": y_pred})
    data_df = data_df.set_index("domain")
    feature_df = pd.melt(
        df,
        value_vars=(df if not absolute else df.abs()).sum(axis=0).sort_values().index[-n:],
        var_name="ngram",
        value_name="SHAP",
        ignore_index=False,
    ).dropna()
    for i, ngram in enumerate(feature_df["ngram"].unique(), start=1):
        feature_df.loc[feature_df["ngram"] == ngram, "central"] = i * 2
    feature_df["value"] = (feature_df["central"] - 0.5) + (np.random.rand(len(feature_df)) * 1)
    feature_df = pd.merge(feature_df, data_df, how="left", left_index=True, right_index=True)
    feature_df = feature_df.reset_index(drop=False)
    fig = px.scatter(
        feature_df,
        x="SHAP",
        y="value",
        color="SHAP",
        color_continuous_scale="Bluered",
        hover_data={
            "SHAP": ":.5f",
            "ngram": True,
            "value": False,
            "domain": True,
            "expected": True,
            "predicted": True,
        },
    )
    fig.update_layout(
        width=1920,
        height=1080,
        xaxis=dict(
            showgrid=True,
            gridcolor="WhiteSmoke",
            zerolinecolor="Black",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="WhiteSmoke",
            zeroline=False,
            tickvals=feature_df["central"].unique(),
            ticktext=feature_df["ngram"].unique(),
            title=dict(
                text="ngram",
            ),
        ),
        plot_bgcolor="white",
    )
    return fig


def feature_plot_features(
    shap_values_csv: Path,
    explainable_npz: Path,
    encoder_p: Path,
    absolute: bool = False,
) -> go.Figure:
    n = 20
    with encoder_p.open("rb") as stream:
        encoder: OneHotEncoder = pickle.load(stream)
    explainable_data = np.load(explainable_npz, allow_pickle=True)
    X = explainable_data["X"]
    y_true = encoder.inverse_transform(explainable_data["y_true"]).reshape([-1])
    y_pred = encoder.inverse_transform(explainable_data["y_pred"]).reshape([-1])
    shap_values_df = pd.read_csv(shap_values_csv, index_col=0)
    values_df = pd.DataFrame(X[:, :-1], columns=shap_values_df.columns)
    rest_df = pd.DataFrame({"domain": X[:, -1], "expected": y_true, "predicted": y_pred})
    feature_df = pd.concat([shap_values_df, values_df], axis=1, keys=["SHAP", "value"])
    feature_df.columns.names = ["", "feature"]
    top_n_features = (
        (feature_df["SHAP"] if not absolute else feature_df["SHAP"].abs()).sum(axis=0).sort_values().index[-n:]
    )
    feature_df = feature_df.loc[:, pd.IndexSlice[:, top_n_features]]
    feature_df = feature_df.stack(level="feature").reset_index(level="feature")
    feature_df = pd.merge(feature_df, rest_df, left_index=True, right_index=True)
    for i, feature in enumerate(top_n_features, start=1):
        feature_df.loc[feature_df["feature"] == feature, "central"] = i * 2
    feature_df["y_value"] = (feature_df["central"] - 0.5) + (np.random.rand(len(feature_df)) * 1)
    fig = go.Figure()
    for feature in top_n_features:
        tmp_df = feature_df[feature_df["feature"] == feature]
        fig.add_trace(
            go.Scatter(
                x=tmp_df["SHAP"],
                y=tmp_df["y_value"],
                text=None,
                name=feature,
                customdata=tmp_df[["SHAP", "value", "domain", "expected", "predicted"]].values,
                hovertemplate="SHAP=%{customdata[0]:.5f}<br>Value=%{customdata[1]}<br>Domain=%{customdata[2]}<br>Expected=%{customdata[3]}<br>Predicted=%{customdata[4]}<extra></extra>",
                mode="markers",
                marker=dict(
                    color=tmp_df["value"],
                    colorscale="Bluered",
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            hoverinfo=None,
            marker=dict(
                showscale=True,
                cmin=0,
                cmax=2,
                colorscale="Bluered",
                colorbar=dict(
                    tickmode="array",
                    tickvals=[0, 2],
                    ticktext=["Low", "High"],
                ),
            ),
        )
    )
    fig.update_layout(
        width=1920,
        height=1080,
        xaxis=dict(
            showgrid=True,
            gridcolor="WhiteSmoke",
            zerolinecolor="Black",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="WhiteSmoke",
            zeroline=False,
            tickvals=feature_df["central"].unique(),
            ticktext=feature_df["feature"].unique(),
            title=dict(
                text="Feature",
            ),
        ),
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


def force_plot_features(
    expected_values_npz: Path,
    shap_values_csv: Path,
    explainable_npz: Path,
    encoder_p: Path,
) -> list[dict[str, Any]]:
    with encoder_p.open("rb") as stream:
        encoder: OneHotEncoder = pickle.load(stream)
    expected_value = np.load(expected_values_npz, allow_pickle=True)["arr"][1]
    explainable_data = np.load(explainable_npz, allow_pickle=True)
    X = explainable_data["X"]
    y_true = encoder.inverse_transform(explainable_data["y_true"]).reshape([-1])
    y_pred = encoder.inverse_transform(explainable_data["y_pred"]).reshape([-1])
    shap_values_df = pd.read_csv(shap_values_csv, index_col=0)
    dfy = pd.DataFrame({"expected": y_true, "predicted": y_pred})
    dfy.loc[dfy[(dfy["expected"] == 0) & (dfy["predicted"] == 0)].index, "label"] = "tn"
    dfy.loc[dfy[(dfy["expected"] == 0) & (dfy["predicted"] == 1)].index, "label"] = "fp"
    dfy.loc[dfy[(dfy["expected"] == 1) & (dfy["predicted"] == 0)].index, "label"] = "fn"
    dfy.loc[dfy[(dfy["expected"] == 1) & (dfy["predicted"] == 1)].index, "label"] = "tp"
    dfy = dfy[["label"]]
    plots = []
    for label in dfy["label"].unique():
        # find first example for this label
        index = dfy[dfy["label"] == label].index.values[0]
        domain = X[index, -1]
        force_plot = shap.force_plot(
            expected_value,
            shap_values_df.iloc[index].fillna(0).values,
            features=X[index, :-1],
            feature_names=shap_values_df.columns.values,
        )
        plots.append({"domain": domain, "label": label, "plot": force_plot})
    return plots
