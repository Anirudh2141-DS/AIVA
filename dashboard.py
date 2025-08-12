import os
import json
import time
import queue
import socket
import threading
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import iframe as st_iframe
import gradio as gr
def find_free_port(default=7861):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", default))
            return default
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
@st.cache_data
def generate_synthetic_data(n=4000, n_features=16, class_sep=1.5, random_state=42):
    X, y = make_classification(
        n_samples=n, n_features=n_features, n_informative=int(n_features*0.6),
        n_redundant=int(n_features*0.2), n_repeated=0, n_classes=2,
        weights=[0.55, 0.45], class_sep=class_sep, flip_y=0.02, random_state=random_state
    )
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    return df
@st.cache_resource
def init_model(model_type="RandomForest", params=None):
    params = params or {}
    if model_type == "RandomForest":
        return RandomForestClassifier(**params)
    return RandomForestClassifier(**params)
def train_and_eval(model, df, test_size=0.2, random_state=42):
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(yte, preds)),
        "roc_auc": float(roc_auc_score(yte, proba)),
        "confusion_matrix": confusion_matrix(yte, preds).tolist(),
        "report": classification_report(yte, preds, output_dict=True)
    }
    return model, (Xte, yte, proba, preds, metrics)
def run_agent_simulation(model, df, n_agents=12, threshold=0.5, generations=5, seed=42):
    """
    Minimal placeholder: each 'agent' uses the same model but different thresholds,
    we evolve thresholds via simple mutation to maximize accuracy on a holdout.
    Replace with your PED loop + GA from AIVA.
    """
    rng = np.random.RandomState(seed)
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)
    split = int(0.7 * len(df))
    X_eval, y_eval = X.iloc[split:], y.iloc[split:]
    population = np.clip(rng.normal(loc=threshold, scale=0.1, size=n_agents), 0.05, 0.95)
    hist = []
    probs = model.predict_proba(X_eval)[:, 1]
    for g in range(generations):
        scores = []
        for t in population:
            preds = (probs >= t).astype(int)
            acc = accuracy_score(y_eval, preds)
            scores.append(acc)
        scores = np.array(scores)
        hist.append({
            "generation": g,
            "best_threshold": float(population[scores.argmax()]),
            "best_acc": float(scores.max()),
            "mean_acc": float(scores.mean())
        })
        elite_idx = scores.argsort()[::-1][: max(2, int(0.3 * n_agents))]
        elite = population[elite_idx]
        children = []
        while len(children) + len(elite) < n_agents:
            p = rng.choice(elite)
            child = np.clip(p + rng.normal(scale=0.05), 0.05, 0.95)
            children.append(child)
        population = np.concatenate([elite, np.array(children)])
    final_scores = []
    for t in population:
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_eval, preds)
        final_scores.append(acc)
    final_scores = np.array(final_scores)
    best_t = float(population[final_scores.argmax()])
    best_acc = float(final_scores.max())
    return {
        "history": hist,
        "best_threshold": best_t,
        "best_acc": best_acc
    }
def build_gradio_app(shared_queue: queue.Queue, feature_names):
    """
    A small predictor/explainer pad:
    - JSON or key/value inputs
    - Returns probability + class
    - Can push events to Streamlit via a queue
    """
    def predict_one(payload_json):
        try:
            d = json.loads(payload_json)
        except Exception:
            return "Invalid JSON. Example: {\"feat_0\": 0.12, \"feat_1\": -1.3, ...}", ""
        x = np.array([[d.get(k, 0.0) for k in feature_names]], dtype=float)
        model_snapshot = None
        try:
            while True:
                model_snapshot = shared_queue.get_nowait()
        except queue.Empty:
            pass
        if model_snapshot is None:
            return "Model not ready. Train in Streamlit first.", ""
        proba = float(model_snapshot.predict_proba(x)[:, 1][0])
        pred = int(proba >= 0.5)
        return f"prob={proba:.4f}", "Invest" if pred == 1 else "Pass"
    with gr.Blocks(title="AIVA Gradio Panel") as demo:
        gr.Markdown("## AIVA – Quick Predictor\nPaste a JSON of feature values to score.")
        example = {f: 0.0 for f in feature_names[:10]}
        inp = gr.Textbox(
            value=json.dumps(example, indent=2),
            lines=10,
            label="Feature JSON"
        )
        out1 = gr.Textbox(label="Probability")
        out2 = gr.Textbox(label="Decision")
        btn = gr.Button("Predict")
        btn.click(predict_one, inputs=inp, outputs=[out1, out2])
    return demo
def launch_gradio_in_thread(blocks: gr.Blocks, port: int):
    def _run():
        blocks.launch(
            server_name="127.0.0.1",
            server_port=port,
            inbrowser=False,
            share=False,
            prevent_thread_lock=True,
            show_error=True
        )
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    time.sleep(0.6)
st.set_page_config(page_title="AIVA Dashboard (Streamlit + Gradio)", layout="wide")
st.title("AIVA Dashboard – Streamlit + Gradio")
st.caption("Hybrid UI: Streamlit for control & telemetry; Gradio for fast prediction pad. Built to slot into your PED/GA pipeline.")
with st.sidebar:
    st.header("Controls")
    n_rows = st.slider("Synthetic rows", 1000, 20000, 6000, 1000)
    n_features = st.slider("Features", 8, 64, 20, 2)
    class_sep = st.slider("Class separability", 0.5, 3.0, 1.6, 0.1)
    rf_trees = st.slider("RandomForest trees", 50, 800, 250, 50)
    rf_depth = st.slider("Max depth (0=auto)", 0, 40, 0, 1)
    sim_agents = st.slider("Agents", 4, 40, 16, 1)
    sim_gens = st.slider("Generations", 1, 30, 8, 1)
    base_thresh = st.slider("Base threshold", 0.05, 0.95, 0.50, 0.05)
    run_train = st.button("Train / Retrain")
    run_sim = st.button("Run Agent Simulation")
df = generate_synthetic_data(n=n_rows, n_features=n_features, class_sep=class_sep)
feature_names = [c for c in df.columns if c != "label"]
params = dict(
    n_estimators=rf_trees,
    max_depth=None if rf_depth == 0 else rf_depth,
    n_jobs=-1,
    random_state=42
)
if "model" not in st.session_state:
    st.session_state.model = init_model("RandomForest", params)
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
if run_train:
    st.session_state.model = init_model("RandomForest", params)
    st.session_state.model, eval_bundle = train_and_eval(st.session_state.model, df)
    Xte, yte, proba, preds, metrics = eval_bundle
    st.session_state.last_metrics = metrics
    st.success("Model trained.")
tab_overview, tab_metrics, tab_sim, tab_gradio = st.tabs(["Overview", "Metrics", "Agent Sim", "Gradio Panel"])
with tab_overview:
    st.subheader("Data Snapshot")
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"Rows: **{len(df):,}**, Features: **{len(feature_names)}**")
with tab_metrics:
    st.subheader("Model Metrics")
    if st.session_state.last_metrics is None:
        st.info("Train the model first.")
    else:
        m = st.session_state.last_metrics
        cmat = np.array(m["confusion_matrix"])
        acc = m["accuracy"]
        roc = m["roc_auc"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("ROC AUC", f"{roc:.3f}")
        col3.metric("Samples", f"{len(df):,}")
        fig = go.Figure(data=go.Heatmap(
            z=cmat,
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            zauto=True
        ))
        fig.update_layout(title="Confusion Matrix", xaxis_nticks=2, yaxis_nticks=2, height=450)
        st.plotly_chart(fig, use_container_width=True)
        rep_df = pd.DataFrame(m["report"]).T
        st.dataframe(rep_df.round(3), use_container_width=True)
with tab_sim:
    st.subheader("Multi-Agent Simulation (toy)")
    if st.session_state.last_metrics is None:
        st.info("Train the model first, then run the simulation.")
    else:
        if run_sim:
            sim = run_agent_simulation(
                st.session_state.model, df,
                n_agents=sim_agents,
                threshold=base_thresh,
                generations=sim_gens
            )
            st.session_state.sim_result = sim
            st.success("Simulation complete.")
        if st.session_state.sim_result:
            sim = st.session_state.sim_result
            st.write(f"**Best Agent Threshold**: {sim['best_threshold']:.3f}  |  **Best Accuracy**: {sim['best_acc']:.3f}")
            hist_df = pd.DataFrame(sim["history"])
            st.dataframe(hist_df, use_container_width=True)
            fig2 = px.line(hist_df, x="generation", y=["best_acc", "mean_acc"], markers=True,
                           title="Agent Fitness (Accuracy) by Generation")
            st.plotly_chart(fig2, use_container_width=True)
with tab_gradio:
    st.subheader("Interactive Predictor (Gradio)")
    st.caption("Train a model in the Metrics tab, then score custom payloads here.")
    if "gradio_queue" not in st.session_state:
        st.session_state.gradio_queue = queue.Queue()
    try:
        while True:
            st.session_state.gradio_queue.get_nowait()
    except queue.Empty:
        pass
    st.session_state.gradio_queue.put(st.session_state.model)
    if "gradio_port" not in st.session_state:
        st.session_state.gradio_port = find_free_port(7861)
    if "gradio_started" not in st.session_state:
        demo = build_gradio_app(st.session_state.gradio_queue, feature_names)
        launch_gradio_in_thread(demo, st.session_state.gradio_port)
        st.session_state.gradio_started = True
    st.info("If you get a blank panel for a moment, give it a second to boot.")
    st_iframe(f"http://127.0.0.1:{st.session_state.gradio_port}", height=640)