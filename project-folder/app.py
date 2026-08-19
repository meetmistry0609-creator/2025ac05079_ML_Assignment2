"""
Dry Bean Classification 
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

# Page configuration

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_TEST_CSV = BASE_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

RAW_COLUMN_RENAME = {"AspectRation": "AspectRatio", "roundness": "Roundness"}

VARIETY_COLORS = {
    "BARBUNYA": "#8B5E3C",
    "BOMBAY":   "#C9A66B",
    "CALI":     "#A23E2E",
    "DERMASON": "#C08A28",
    "HOROZ":    "#55694A",
    "SEKER":    "#3E4F36",
    "SIRA":     "#7C6A46",
}

BEAN_CMAP = LinearSegmentedColormap.from_list("bean", ["#F7F1E3", "#C08A28", "#3E4F36"])


def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root {
            --paper: #EDE4D2;
            --panel: #F7F1E3;
            --ink: #2B2118;
            --ink-soft: #6B5B48;
            --pod: #3E4F36;
            --pod-soft: #55694A;
            --gold: #C08A28;
            --brick: #A23E2E;
            --border: #D8C9A3;
        }

        .stApp { background-color: var(--paper); }
        section[data-testid="stSidebar"] { background-color: var(--panel); border-right: 1px solid var(--border); }

        h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--ink) !important; letter-spacing: -0.01em; }
        p, li, span, label, div { font-family: 'IBM Plex Sans', sans-serif; }
        .stCaption, [data-testid="stCaptionContainer"] { font-family: 'IBM Plex Sans', sans-serif !important; color: var(--ink-soft) !important; }

        /* Hero banner */
        .bean-hero {
            background: linear-gradient(135deg, var(--pod) 0%, var(--pod-soft) 100%);
            border-radius: 18px;
            padding: 2.1rem 2.4rem;
            margin-bottom: 1.6rem;
            color: #F7F1E3;
        }
        .bean-hero__eyebrow {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #D8C9A3;
            margin-bottom: 0.5rem;
        }
        .bean-hero__title {
            font-family: 'Fraunces', serif;
            font-size: 2.3rem;
            font-weight: 600;
            margin: 0 0 0.4rem 0;
            color: #FBF8F0;
        }
        .bean-hero__sub {
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 1rem;
            color: #E7DFC9;
            max-width: 62ch;
            margin: 0;
        }
        .bean-row { display: flex; gap: 0.55rem; margin-top: 1.1rem; flex-wrap: wrap; }
        .bean-chip {
            display: inline-flex; align-items: center; gap: 0.4rem;
            background: rgba(247,241,227,0.12);
            border: 1px solid rgba(247,241,227,0.28);
            border-radius: 999px;
            padding: 0.28rem 0.7rem 0.28rem 0.5rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.74rem;
            color: #FBF8F0;
        }
        .bean-dot { width: 11px; height: 15px; border-radius: 50% 50% 48% 48% / 60% 60% 40% 40%; display: inline-block; }

        /* Section labels */
        .bean-eyebrow {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--gold);
            margin-bottom: -0.4rem;
        }

        /* Callout card (best model, warnings) */
        .bean-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-left: 4px solid var(--gold);
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin: 0.6rem 0 1rem 0;
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--ink);
        }
        .bean-card b { font-family: 'IBM Plex Mono', monospace; color: var(--pod); }

        /* Metric tiles (st.metric override) */
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.7rem 0.9rem 0.5rem 0.9rem;
        }
        [data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace !important; color: var(--ink-soft) !important; }
        [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; color: var(--pod) !important; }

        /* Dataframes */
        [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

        hr { border-color: var(--border) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    chips = "".join(
        f'<span class="bean-chip"><span class="bean-dot" style="background:{color};"></span>{name.title()}</span>'
        for name, color in VARIETY_COLORS.items()
    )
    st.markdown(
        f"""
        <div class="bean-hero">
            <div class="bean-hero__eyebrow">ML Assignment 2 &middot; UCI Dry Bean Dataset</div>
            <div class="bean-hero__title">🫘 Dry Bean Classification</div>
            <p class="bean-hero__sub">Seven registered varieties, nineteen shape and size measurements
            pulled from bean photographs, five classical classifiers trained on the same split.
            Shape is the data for every prediction below comes from geometry alone.</p>
            <div class="bean-row">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Cached loaders

@st.cache_resource
def load_models():
    models = {}
    for name, fname in MODEL_FILES.items():
        path = MODEL_DIR / fname
        with open(path, "rb") as fh:
            models[name] = pickle.load(fh)
    return models


@st.cache_resource
def load_metadata():
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as fh:
        label_encoder = pickle.load(fh)
    with open(MODEL_DIR / "feature_columns.pkl", "rb") as fh:
        feature_cols = pickle.load(fh)
    return label_encoder, feature_cols


@st.cache_data
def load_default_test_data():
    return pd.read_csv(DEFAULT_TEST_CSV)

# Feature engineering & label handling

def prepare_features(raw_df: pd.DataFrame, feature_cols: list[str]):
    """Clean column names, engineer the 3 ratio features, and align to the
    exact feature set the models were trained on. Returns (X, df_with_class)."""
    df = raw_df.copy()
    df = df.rename(columns=RAW_COLUMN_RENAME)
    if "ConvexityDeficit" not in df.columns and {"ConvexArea", "Area"}.issubset(df.columns):
        df["ConvexityDeficit"] = (df["ConvexArea"] - df["Area"]) / df["ConvexArea"]
    if "AreaPerimeterRatio" not in df.columns and {"Area", "Perimeter"}.issubset(df.columns):
        df["AreaPerimeterRatio"] = df["Area"] / df["Perimeter"]
    if "DiameterRatio" not in df.columns and {"EquivDiameter", "MajorAxisLength"}.issubset(df.columns):
        df["DiameterRatio"] = df["EquivDiameter"] / df["MajorAxisLength"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required feature column(s): " + ", ".join(missing)
        )

    X = df[feature_cols]
    return X, df


def align_class_labels(series: pd.Series, label_encoder):
    """Match an uploaded 'Class' column to the encoder's fitted class strings
    regardless of case (the trained encoder was fit on raw UCI labels, which
    are upper-case — e.g. 'SEKER', not 'Seker'). Returns (aligned, unmapped_mask):
    aligned holds the encoder's own casing wherever a match was found, and
    unmapped_mask flags rows whose label truly isn't one of the known classes."""
    classes = list(label_encoder.classes_)
    upper_map = {c.upper(): c for c in classes}
    normalized = series.astype(str).str.strip().str.upper()
    aligned = normalized.map(upper_map)
    unmapped_mask = aligned.isna()
    return aligned, unmapped_mask


def compute_metrics(y_true_enc, y_pred_enc, y_proba, n_classes):
    acc = accuracy_score(y_true_enc, y_pred_enc)
    prec = precision_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
    rec = recall_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
    f1 = f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true_enc, y_pred_enc)
    try:
        if n_classes > 2:
            auc = roc_auc_score(y_true_enc, y_proba, multi_class="ovr", average="weighted")
        else:
            auc = roc_auc_score(y_true_enc, y_proba[:, 1])
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": acc, "AUC": auc, "Precision": prec,
        "Recall": rec, "F1": f1, "MCC": mcc,
    }


@st.cache_data(show_spinner=False)
def score_all_models(_models_tuple, X, y_true_enc, n_classes):
    """Run every model once and return a tidy metrics DataFrame. Cached on the
    (hashable) model names so re-runs on the same data/models are instant."""
    rows = []
    for name, pipe in _models_tuple:
        y_pred_enc = pipe.predict(X)
        y_proba = pipe.predict_proba(X)
        m = compute_metrics(y_true_enc, y_pred_enc, y_proba, n_classes)
        rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in m.items()}})
    return pd.DataFrame(rows).set_index("ML Model Name")


def confusion_heatmap(cm, class_names, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
    else:
        fig = ax.figure
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=BEAN_CMAP,
        xticklabels=[c.title() for c in class_names],
        yticklabels=[c.title() for c in class_names],
        linewidths=0.6, linecolor="#EDE4D2",
        cbar=False, ax=ax, annot_kws={"fontsize": 9},
    )
    ax.set_xlabel("Predicted", fontfamily="sans-serif")
    ax.set_ylabel("Actual", fontfamily="sans-serif")
    ax.set_title(title, fontfamily="serif", fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig

# Sidebar data upload and model selection

inject_theme()

st.sidebar.markdown(
    '<div class="bean-eyebrow">Controls</div>'
    '<h3 style="margin-top:0.1rem;">🫘 Dry Bean Classifier</h3>'
    '<p style="color:var(--ink-soft); margin-top:-0.6rem; font-size:0.88rem;">'
    'Model demo &amp; evaluation</p>',
    unsafe_allow_html=True,
)

st.sidebar.header("1. Test data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV of bean measurements", type=["csv"],
    help="Must contain the 16 UCI Dry Bean feature columns. An optional "
         "'Class' column enables evaluation metrics.",
)

st.sidebar.header("2. Model")
model_choice = st.sidebar.selectbox(
    "Focus on one classifier",
    ["Compare all models"] + list(MODEL_FILES.keys()),
    help="Pick a single model for a detailed drill-down, or leave this on "
         "'Compare all models' for a side-by-side view. Either way, the "
         "leaderboard below the data preview always shows every model.",
)

with st.sidebar.expander("About this dataset"):
    st.markdown(
        "**UCI Dry Bean Dataset** — 13,611 images of dry beans across "
        "**7 varieties** (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, "
        "Sira), described by 16 shape/size measurements. 3 extra ratio "
        "features (Convexity Deficit, Area/Perimeter Ratio, Diameter Ratio) "
        "were engineered on top, for 19 features total."
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Models: Logistic Regression · Decision Tree · kNN · Naive Bayes · "
    "Random Forest (Ensemble), all trained on the UCI Dry Bean dataset."
)


# Load models and metadata

try:
    models = load_models()
    label_encoder, feature_cols = load_metadata()
except FileNotFoundError as e:
    st.error(
        "Could not find trained model files in `model/`. Make sure the "
        "repository includes the `.pkl` files produced by the training "
        f"notebook.\n\n{e}"
    )
    st.stop()


# Input dataframe

render_hero()

try:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {raw_df.shape[0]} rows from uploaded file.")
except Exception as e:
    st.error(f"Upload a CSV file (e.g. test_data.csv) using the sidebar to get started.")
    st.stop()
    
with st.expander("Preview input data", expanded=False):
    st.dataframe(raw_df.head(20), use_container_width=True)

has_labels = "Class" in raw_df.columns

try:
    X, df_full = prepare_features(raw_df, feature_cols)
except ValueError as e:
    st.error(str(e))
    st.stop()

y_true_enc = None
if has_labels:
    aligned, unmapped_mask = align_class_labels(df_full["Class"], label_encoder)
    n_unmapped = int(unmapped_mask.sum())
    if n_unmapped == len(df_full):
        st.warning(
            "The 'Class' column contains no labels recognized from training — "
            "evaluation metrics will be skipped, predictions will still be shown."
        )
        has_labels = False
    else:
        if n_unmapped > 0:
            st.warning(
                f"{n_unmapped} of {len(df_full)} rows have a 'Class' value not seen "
                "during training and are excluded from evaluation metrics below "
                "(their predictions still appear in the tables)."
            )
        df_full = df_full.loc[~unmapped_mask].reset_index(drop=True)
        X = X.loc[~unmapped_mask].reset_index(drop=True) if hasattr(X, "loc") else X[~unmapped_mask.values]
        y_true_enc = label_encoder.transform(aligned[~unmapped_mask])

n_classes = len(label_encoder.classes_)

if has_labels:
    st.markdown('<div class="bean-eyebrow">Leaderboard</div>', unsafe_allow_html=True)
    st.subheader("📌 All 5 models on this test data")
    leaderboard = score_all_models(tuple(models.items()), X, y_true_enc, n_classes)
    best_by_mcc = leaderboard["MCC"].idxmax()
    st.dataframe(
        leaderboard.style.highlight_max(axis=0, color="#C08A28").format(precision=4),
        use_container_width=True,
    )
    st.markdown(
        f'<div class="bean-card">Best model on this data by MCC: <b>{best_by_mcc}</b></div>',
        unsafe_allow_html=True,
    )
else:
    st.info(
        "This test data has no **'Class'** column, so evaluation metrics "
        "and the leaderboard are skipped — you'll still see predictions "
        "for every model below."
    )

st.markdown("---")

# Single-model view

def render_single_model(name):
    pipe = models[name]
    y_pred_enc = pipe.predict(X)
    y_proba = pipe.predict_proba(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

    result_df = df_full.copy()
    result_df["Predicted Class"] = [c.title() for c in y_pred_labels]
    result_df["Prediction Confidence"] = y_proba.max(axis=1).round(4)
    if has_labels:
        result_df["Correct"] = result_df["Class"].str.upper() == pd.Series(y_pred_labels, index=result_df.index)

    if not has_labels:
        st.subheader(f"Predictions — {name}")
        cols_to_show = ["Predicted Class", "Prediction Confidence"]
        st.dataframe(result_df[cols_to_show].head(50), use_container_width=True)
        st.download_button(
            "⬇ Download predictions (CSV)",
            result_df[cols_to_show].to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{name.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )
        st.info(
            "Upload a CSV containing a **'Class'** column with ground-truth "
            "labels to see evaluation metrics, a confusion matrix and a "
            "classification report."
        )
        return

    metrics = compute_metrics(y_true_enc, y_pred_enc, y_proba, n_classes)

    tab_pred, tab_metrics, tab_cm, tab_report = st.tabs(
        ["Predictions", "Metrics", "Confusion Matrix", "Classification Report"]
    )

    with tab_pred:
        n_correct = int(result_df["Correct"].sum())
        st.caption(f"{n_correct} / {len(result_df)} correct on this sample "
                   f"({n_correct / len(result_df):.1%})")
        cols_to_show = ["Class", "Predicted Class", "Prediction Confidence", "Correct"]
        st.dataframe(
            result_df[cols_to_show].head(50).style.map(
                lambda v: "color: #A23E2E" if v is False else ("color: #3E4F36" if v is True else ""),
                subset=["Correct"],
            ),
            use_container_width=True,
        )
        st.download_button(
            "⬇ Download predictions (CSV)",
            result_df[cols_to_show].to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{name.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )

    with tab_metrics:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        m2.metric("AUC", f"{metrics['AUC']:.4f}" if not np.isnan(metrics["AUC"]) else "N/A")
        m3.metric("Precision", f"{metrics['Precision']:.4f}")
        m4.metric("Recall", f"{metrics['Recall']:.4f}")
        m5.metric("F1 Score", f"{metrics['F1']:.4f}")
        m6.metric("MCC", f"{metrics['MCC']:.4f}")

    with tab_cm:
        cm = confusion_matrix(y_true_enc, y_pred_enc, labels=range(n_classes))
        fig = confusion_heatmap(cm, label_encoder.classes_, f"Confusion Matrix — {name}")
        st.pyplot(fig)

    with tab_report:
        report_dict = classification_report(
            y_true_enc, y_pred_enc, target_names=[c.title() for c in label_encoder.classes_],
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose().round(3)
        st.dataframe(report_df, use_container_width=True, height=320)

# Compare-all-models view

def render_comparison():
    if not has_labels:
        st.subheader("Predictions of all models")
        preds = {}
        for name, pipe in models.items():
            preds[name] = [c.title() for c in label_encoder.inverse_transform(pipe.predict(X))]
        preds_df = pd.DataFrame(preds)
        st.dataframe(preds_df.head(50), use_container_width=True)
        st.download_button(
            "⬇ Download all predictions (CSV)",
            preds_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions_all_models.csv",
            mime="text/csv",
        )
        return

    tab_chart, tab_cm = st.tabs(["📊 Metric Comparison Chart", "🧩 Best Model's Confusion Matrix"])

    with tab_chart:
        variety_palette = ["#3E4F36", "#C08A28", "#A23E2E", "#8B5E3C", "#55694A", "#7C6A46"]
        fig, ax = plt.subplots(figsize=(10, 5))
        leaderboard.plot(kind="bar", ax=ax, color=variety_palette)
        ax.set_ylabel("Score", fontfamily="sans-serif")
        ax.set_title("Model comparison across metrics", fontfamily="serif", fontsize=13)
        ax.set_facecolor("#F7F1E3")
        fig.patch.set_facecolor("#F7F1E3")
        plt.xticks(rotation=20, ha="right")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with tab_cm:
        best_pipe = models[best_by_mcc]
        y_pred_best = best_pipe.predict(X)
        best_cm = confusion_matrix(y_true_enc, y_pred_best, labels=range(n_classes))
        st.caption(f"Showing **{best_by_mcc}** (highest MCC on this data)")
        fig2 = confusion_heatmap(best_cm, label_encoder.classes_, "", ax=None)
        st.pyplot(fig2)


if model_choice == "Compare all models":
    render_comparison()
else:
    render_single_model(model_choice)

st.markdown("---")