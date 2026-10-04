"""Tumour Risk Predictor - interactive Streamlit app.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import logging
import math
import time

import pandas as pd
import streamlit as st

import predictor

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = 2
st.set_page_config(page_title="Tumour Risk Predictor", page_icon="🔬", layout="wide")


# --------------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return predictor.load_artifacts()


try:
    model, meta = get_artifacts()
except predictor.ModelLoadError as exc:
    st.error(f"The model could not be loaded. {exc}")
    st.stop()

FEATURES = meta["features"]
CLASS_NAMES = meta["class_names"]


# --------------------------------------------------------------------------- helpers
def widget_key(feature_key: str) -> str:
    return f"in_{feature_key}"


def apply_example(name: str) -> None:
    for key, value in meta["examples"][name].items():
        st.session_state[widget_key(key)] = float(value)


def clear_inputs() -> None:
    for feat in FEATURES:
        st.session_state[widget_key(feat["key"])] = None


def step_and_format(feat: dict) -> tuple[float, str]:
    span = feat["max"] - feat["min"]
    step = 10 ** math.floor(math.log10(span / 100)) if span > 0 else 0.01
    decimals = max(0, -int(math.floor(math.log10(step))))
    return float(step), f"%.{min(decimals + 1, 6)}f"


def risk_band(p: float) -> tuple[str, str]:
    """(band name, which Streamlit status box to use). Text always accompanies colour."""
    if p < 0.2:
        return "Low risk", "success"
    if p < 0.5:
        return "Moderate risk", "info"
    if p < 0.8:
        return "Elevated risk", "warning"
    return "High risk", "error"


def render_result(pred: predictor.Prediction, contrib: pd.DataFrame | None, elapsed_ms: float) -> None:
    band, kind = risk_band(pred.probability_malignant)
    getattr(st, kind)(
        f"**{pred.label}** - {band} "
        f"(probability of malignancy: {pred.probability_malignant:.1%})"
    )
    for warning in pred.warnings:
        st.warning(warning)

    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction", pred.label)
    m2.metric("Confidence", f"{pred.confidence:.1%}")
    m3.metric("Inference time", f"{elapsed_ms:.1f} ms")

    st.progress(
        min(max(pred.probability_malignant, 0.0), 1.0),
        text=f"Risk meter: {pred.probability_malignant:.1%} probability of malignancy",
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Class probabilities**")
        st.bar_chart(pd.DataFrame({"Probability": pred.probabilities}))
    with right:
        if contrib is not None:
            st.markdown("**What drove this prediction**")
            st.caption("Contribution to the log-odds of *Malignant*. Positive pushes toward malignant, negative toward benign.")
            st.bar_chart(contrib.set_index("Feature")["Contribution"])


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Open **Single prediction** and type the eight nucleus measurements, "
        "or click an example profile to fill them in.\n"
        "2. Press **Predict** to see the probability, risk meter and the features behind it.\n"
        "3. Have many samples? Use **Batch (CSV)** and download the template to see the format."
    )
    st.divider()
    st.caption(
        "⚠️ Educational demo trained on a small public dataset. It is **not** a medical "
        "device and must not be used for diagnosis."
    )

# --------------------------------------------------------------------------- main
st.title("🔬 Tumour Risk Predictor")
st.write(
    "Estimates whether a breast-mass cell nucleus is **benign** or **malignant** from eight "
    "measurements, using a model trained on the Breast Cancer Wisconsin (Diagnostic) dataset."
)

tab_single, tab_batch, tab_about = st.tabs(["Single prediction", "Batch (CSV)", "About the model"])

# ---- single prediction
with tab_single:
    st.subheader("Enter measurements")
    st.caption("Start from an example profile, or type your own values. All fields are required.")

    b1, b2, b3 = st.columns(3)
    example_names = list(meta["examples"])
    b1.button(example_names[0], on_click=apply_example, args=(example_names[0],), use_container_width=True)
    b2.button(example_names[1], on_click=apply_example, args=(example_names[1],), use_container_width=True)
    b3.button("Clear all fields", on_click=clear_inputs, use_container_width=True)

    with st.form("single_prediction"):
        cols = st.columns(2)
        for i, feat in enumerate(FEATURES):
            step, fmt = step_and_format(feat)
            with cols[i % 2]:
                st.number_input(
                    feat["label"],
                    min_value=0.0,
                    value=None,
                    step=step,
                    format=fmt,
                    key=widget_key(feat["key"]),
                    placeholder=f"{feat['min']:g} to {feat['max']:g}",
                    help=f"{feat['help']} Typical range in training data: {feat['min']:g} to {feat['max']:g}.",
                )
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        raw = {f["key"]: st.session_state.get(widget_key(f["key"])) for f in FEATURES}
        try:
            start = time.perf_counter()
            prediction = predictor.predict(model, meta, raw)
            contrib = predictor.contributions(model, meta, raw)
            elapsed = (time.perf_counter() - start) * 1000
        except predictor.ValidationError as exc:
            st.error("Please fix the following before predicting:")
            for message in exc.errors:
                st.markdown(f"- {message}")
        except Exception:  # noqa: BLE001 - never show a stack trace to the user
            logger.exception("Prediction failed")
            st.error("Something went wrong while predicting. Please check your inputs and try again.")
        else:
            render_result(prediction, contrib, elapsed)

# ---- batch prediction
with tab_batch:
    st.subheader("Predict many samples from a CSV")
    st.markdown(
        f"Upload a `.csv` file (max {MAX_UPLOAD_MB} MB, {predictor.MAX_BATCH_ROWS} rows) with these columns:"
    )
    st.code(", ".join(predictor.feature_keys(meta)), language=None)

    template = pd.DataFrame(list(meta["examples"].values()))[predictor.feature_keys(meta)]
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Only .csv files are supported.")
    if uploaded is not None:
        if not uploaded.name.lower().endswith(".csv"):
            st.error("Unsupported file type. Please upload a .csv file.")
        elif uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is too large ({uploaded.size / 1024 / 1024:.1f} MB). The limit is {MAX_UPLOAD_MB} MB.")
        else:
            try:
                frame = pd.read_csv(uploaded)
                result = predictor.predict_dataframe(model, meta, frame)
            except pd.errors.EmptyDataError:
                st.error("The file is empty.")
            except (pd.errors.ParserError, UnicodeDecodeError):
                st.error("Could not read the file as CSV. Check that it is comma-separated, UTF-8 text.")
            except predictor.ValidationError as exc:
                for message in exc.errors:
                    st.error(message)
            except Exception:  # noqa: BLE001
                logger.exception("Batch prediction failed")
                st.error("Something went wrong while processing the file.")
            else:
                ok = result["prediction"].notna()
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows processed", int(ok.sum()))
                c2.metric("Predicted malignant", int((result["prediction"] == "Malignant").sum()))
                c3.metric("Rows with errors", int((~ok).sum()))
                if (~ok).any():
                    st.warning("Some rows could not be scored - see the *note* column (row numbers match your file).")
                st.dataframe(result, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download results",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

# ---- about
with tab_about:
    st.subheader("About the model")
    tm = meta["test_metrics"]
    st.markdown(
        f"- **Dataset:** {meta['dataset']} ({meta['n_samples']} samples, bundled with scikit-learn)\n"
        f"- **Model:** {meta['model_type'].replace('_', ' ').title()} on standardised features\n"
        f"- **Target:** probability the sample is *{meta['positive_class']}*\n"
        f"- **Trained with:** scikit-learn {meta['trained_with']['scikit_learn']}"
    )
    st.markdown("**Held-out test performance (20% split, never used for model selection)**")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Accuracy", f"{tm['accuracy']:.1%}")
    t2.metric("Precision", f"{tm['precision']:.1%}")
    t3.metric("Recall", f"{tm['recall']:.1%}")
    t4.metric("F1", f"{tm['f1']:.1%}")
    t5.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    cm = tm["confusion_matrix"]
    st.markdown("**Confusion matrix (test set)**")
    st.dataframe(
        pd.DataFrame(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            index=["Actually benign", "Actually malignant"],
            columns=["Predicted benign", "Predicted malignant"],
        ),
        use_container_width=True,
    )
    st.markdown("**Limitations**")
    st.markdown(
        "- Trained on ~570 samples from one source; it may not generalise to other populations or imaging setups.\n"
        "- Uses only eight summary measurements, not images.\n"
        "- Probabilities are model estimates, not clinical risk. Values far outside the training range trigger a warning."
    )
