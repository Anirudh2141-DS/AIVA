import os
import io
import gc
import json
import math
import time
import queue
import socket
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import gradio as gr
def find_free_port(default=7861):
    import socket as _socket
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", default))
            return default
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
@st.cache_data(show_spinner=False)
def gen_synth(n:int, n_features:int, class_sep:float, seed:int=42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n, n_features=n_features,
        n_informative=int(n_features*0.6),
        n_redundant=int(n_features*0.2),
        class_sep=class_sep, flip_y=0.02,
        weights=[0.55, 0.45], random_state=seed
    )
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y.astype(int)
    return df
def _model_picker(name:str, params:Dict):
    name = (name or "RandomForest").lower()
    if name == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(**params)
    if name == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(**params)
    return RandomForestClassifier(**params)
@st.cache_resource(show_spinner=False)
def init_model(alg:str, params:Dict):
    return _model_picker(alg, params)
def train_model(model, df:pd.DataFrame, test_size:float=0.2, seed:int=42):
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:,1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(yte, preds)),
        "roc_auc": float(roc_auc_score(yte, proba)),
        "confusion_matrix": confusion_matrix(yte, preds).tolist(),
        "report": classification_report(yte, preds, output_dict=True),
        "n_test": int(len(yte))
    }
    return model, (Xte, yte, proba, preds, metrics)
def _read_large_csv_chunks(file, chunksize:int=100_000, dtype=None):
    for chunk in pd.read_csv(file, chunksize=chunksize, dtype=dtype):
        yield chunk
def _read_large_parquet(file, columns=None, batch_size:int=200_000):
    df = pd.read_parquet(file, columns=columns)
    n = len(df)
    for i in range(0, n, batch_size):
        yield df.iloc[i:i+batch_size].copy()
def _iter_batches(file, fmt:str, label_col:str, drop_non_numeric:bool=True, chunksize:int=100_000):
    if fmt == "csv":
        for chunk in _read_large_csv_chunks(file, chunksize=chunksize):
            if drop_non_numeric:
                numeric = chunk.select_dtypes(include=[np.number])
                if label_col in chunk.columns and label_col not in numeric.columns:
                    numeric[label_col] = chunk[label_col].values
                chunk = numeric
            chunk = chunk.dropna()
            if label_col not in chunk.columns:
                continue
            y = chunk[label_col].astype(int)
            X = chunk.drop(columns=[label_col])
            yield X, y
    else:
        for chunk in _read_large_parquet(file):
            if drop_non_numeric:
                numeric = chunk.select_dtypes(include=[np.number])
                if label_col in chunk.columns and label_col not in numeric.columns:
                    numeric[label_col] = chunk[label_col].values
                chunk = numeric
            chunk = chunk.dropna()
            if label_col not in chunk.columns:
                continue
            y = chunk[label_col].astype(int)
            X = chunk.drop(columns=[label_col])
            yield X, y
@st.cache_data(show_spinner=False)
def quick_profile_sample(file_bytes:bytes, fmt:str, max_rows:int=5000) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if fmt == "csv":
        return pd.read_csv(bio, nrows=max_rows)
    else:
        return pd.read_parquet(bio)
def evaluate_on_large_dataset(model, file_bytes:bytes, fmt:str, label_col:str, threshold:float=0.5,
                              chunksize:int=100_000) -> Dict:
    bio_for_iter = io.BytesIO(file_bytes)
    proba_all = []
    y_all = []
    n_total = 0
    iterator = _iter_batches(bio_for_iter, fmt=fmt, label_col=label_col, chunksize=chunksize)
    for Xb, yb in iterator:
        probs = model.predict_proba(Xb)[:,1]
        proba_all.append(probs)
        y_all.append(yb.values)
        n_total += len(yb)
    if n_total == 0:
        return {"error":"No rows detected with given label column."}
    y = np.concatenate(y_all)
    proba = np.concatenate(proba_all)
    preds = (proba >= threshold).astype(int)
    metrics = {
        "n_rows": int(n_total),
        "accuracy": float(accuracy_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "report": classification_report(y, preds, output_dict=True),
    }
    failures_mask = (preds != y)
    return {
        "y": y, "proba": proba, "preds": preds,
        "fail_mask": failures_mask,
        "metrics": metrics
    }
def run_multi_agent(model, X_eval:pd.DataFrame, y_eval:pd.Series,
                    n_agents:int=16, generations:int=8, seed:int=42,
                    base_threshold:float=0.5) -> Dict:
    rng = np.random.RandomState(seed)
    probs = model.predict_proba(X_eval)[:,1]
    pop = np.clip(rng.normal(loc=base_threshold, scale=0.1, size=n_agents), 0.05, 0.95)
    risk = np.clip(rng.beta(a=3, b=3, size=n_agents), 0.1, 0.9)  # 0=conservative, 1=aggressive
    def fitness(thresh, r):
        t = np.clip(thresh + (r - 0.5)*0.1, 0.01, 0.99)
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_eval, preds)
        ent = -(np.mean(probs*np.log(np.clip(probs,1e-8,1))) + np.mean((1-probs)*np.log(np.clip(1-probs,1e-8,1))))
        return acc + 0.01*ent
    history = []
    for g in range(generations):
        scores = np.array([fitness(pop[i], risk[i]) for i in range(n_agents)])
        best_idx = int(scores.argmax())
        preds_best = (probs >= np.clip(pop[best_idx]+(risk[best_idx]-0.5)*0.1,0.01,0.99)).astype(int)
        history.append({
            "generation": g,
            "best_threshold": float(pop[best_idx]),
            "best_acc": float(accuracy_score(y_eval, preds_best)),
            "mean_acc": float(np.mean(scores))
        })
        elite_idx = scores.argsort()[::-1][:max(2, int(0.3*n_agents))]
        elite_pop = pop[elite_idx]
        elite_risk = risk[elite_idx]
        children_t, children_r = [], []
        while len(children_t)+len(elite_pop) < n_agents:
            i, j = rng.choice(len(elite_pop), size=2, replace=True)
            t_child = np.clip((elite_pop[i]+elite_pop[j])/2 + rng.normal(0,0.05), 0.05, 0.95)
            r_child = np.clip((elite_risk[i]+elite_risk[j])/2 + rng.normal(0,0.05), 0.05, 0.95)
            children_t.append(t_child)
            children_r.append(r_child)
        pop = np.concatenate([elite_pop, np.array(children_t)])
        risk = np.concatenate([elite_risk, np.array(children_r)])
    final_scores = np.array([fitness(pop[i], risk[i]) for i in range(n_agents)])
    best = int(final_scores.argmax())
    best_t, best_r = float(pop[best]), float(risk[best])
    preds_final = (probs >= np.clip(best_t + (best_r-0.5)*0.1, 0.01, 0.99)).astype(int)
    best_acc = float(accuracy_score(y_eval, preds_final))
    return {"history": history, "best_threshold": best_t, "best_risk": best_r, "best_acc": best_acc}
def gather_failures(X_test:pd.DataFrame, y_test:pd.Series, proba:np.ndarray, preds:np.ndarray,
                    top_k:int=0) -> Tuple[pd.DataFrame, pd.Series]:
    wrong = preds != y_test.values
    if top_k and top_k > 0:
        margin = np.abs(proba - y_test.values)
        idx_wrong = np.where(wrong)[0]
        if len(idx_wrong) > top_k:
            sel = idx_wrong[np.argsort(margin[idx_wrong])[::-1][:top_k]]
        else:
            sel = idx_wrong
        return X_test.iloc[sel].copy(), y_test.iloc[sel].copy()
    return X_test.iloc[wrong].copy(), y_test.iloc[wrong].copy()
def failure_retrain(base_model, base_train_df:pd.DataFrame,
                    fail_sets:List[Tuple[pd.DataFrame, pd.Series]],
                    weight_failures:float=2.0, seed:int=42):
    X_base = base_train_df.drop(columns=["label"])
    y_base = base_train_df["label"].astype(int)
    X_fails = []
    y_fails = []
    for Xf, yf in fail_sets:
        if len(yf) == 0:
            continue
        X_fails.append(Xf)
        y_fails.append(yf.astype(int))
    if len(y_fails) == 0:
        model = _model_picker(type(base_model).__name__, base_model.get_params())
        model.fit(X_base, y_base)
        return model
    Xf = pd.concat(X_fails, axis=0)
    yf = pd.concat(y_fails, axis=0).astype(int)
    rep = max(1, int(weight_failures))
    X_aug = pd.concat([X_base, pd.concat([Xf]*rep)], axis=0, ignore_index=True)
    y_aug = pd.concat([y_base, pd.concat([yf]*rep)], axis=0, ignore_index=True)
    model = _model_picker(type(base_model).__name__, base_model.get_params())
    model.random_state = getattr(base_model, "random_state", seed)
    model.fit(X_aug, y_aug)
    return model
def build_gradio(shared_queue:queue.Queue, feature_names:List[str]):
    def predict_one(payload):
        try:
            d = json.loads(payload)
        except Exception:
            return "Invalid JSON", ""
        x = np.array([[d.get(k, 0.0) for k in feature_names]], dtype=float)
        latest = None
        try:
            while True:
                latest = shared_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is None:
            return "Model not ready", ""
        proba = float(latest.predict_proba(x)[:,1][0])
        pred = int(proba >= 0.5)
        return f"{proba:.4f}", ("Invest" if pred==1 else "Pass")
    with gr.Blocks(title="AIVA – Quick Predictor") as demo:
        gr.Markdown("### AIVA – Quick Predictor\nPaste feature JSON to score.")
        example = {f"feat_{i}":0.0 for i in range(min(len(feature_names), 12))}
        inp = gr.Textbox(value=json.dumps(example, indent=2), lines=12, label="Feature JSON")
        out1 = gr.Textbox(label="Probability")
        out2 = gr.Textbox(label="Decision")
        btn = gr.Button("Predict")
        btn.click(predict_one, inputs=inp, outputs=[out1, out2])
    return demo
def launch_gradio(blocks:gr.Blocks, port:int):
    def _run():
        blocks.launch(server_name="127.0.0.1", server_port=port, inbrowser=False,
                      share=False, prevent_thread_lock=True, show_error=True)
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    time.sleep(0.6)
st.set_page_config(page_title="AIVA – Full Phases 1–4", layout="wide")
st.title("AIVA Dashboard – Full PED Phases 1–4")
st.caption("Phase 2 uses a real dataset (~1.1M rows): CSV or Parquet. No scraping.")
with st.sidebar:
    st.header("Controls")
    n_rows   = st.slider("Synthetic rows", 2000, 20000, 6000, step=1000)
    n_feats  = st.slider("Features", 6, 64, 20, step=2)
    sep      = st.slider("Class separability", 0.8, 3.0, 1.6, step=0.05)
    alg      = st.selectbox("Algorithm", ["RandomForest", "XGBoost", "LightGBM"])
    trees    = st.slider("Trees/Estimators", 50, 600, 250, step=25)
    max_depth= st.slider("Max depth (0=auto)", 0, 32, 0, step=1)
    test_size= st.slider("Test split", 0.1, 0.4, 0.2, step=0.05)
    seed     = st.number_input("Seed", value=42, step=1)
    st.markdown("---")
    st.subheader("Agent Sim")
    n_agents = st.slider("Agents", 4, 64, 16, step=2)
    gens     = st.slider("Generations", 2, 30, 8, step=1)
    base_t   = st.slider("Base threshold", 0.05, 0.95, 0.50, step=0.01)
    st.markdown("---")
    st.subheader("Phase 2 (Real data)")
    label_col = st.text_input("Label column name", value="label")
    real_threshold = st.slider("Decision threshold (Phase 2)", 0.05, 0.95, 0.5, step=0.01)
    chunk = st.slider("CSV chunk size", 50_000, 300_000, 100_000, step=50_000)
tabs = st.tabs(["Overview", "Phase 1: Synthetic Train", "Phase 2: Real Eval",
                "Phase 3: Multi-Agent GA", "Phase 4: Failure Retrain", "Gradio Pad"])
with tabs[1]:
    st.subheader("Generate Synthetic & Train Baseline")
    df = gen_synth(n_rows, n_feats, sep, seed)
    st.success("Synthetic data ready.")
    st.dataframe(df.head(20), use_container_width=True)
    params = {"n_estimators": trees, "random_state": seed}
    if max_depth > 0 and alg.lower()=="randomforest":
        params["max_depth"] = max_depth
    if alg.lower()=="xgboost" and XGBClassifier is not None:
        params = {"n_estimators":trees, "learning_rate":0.1, "max_depth": (max_depth or 6),
                  "subsample":0.9, "colsample_bytree":0.9, "random_state":seed, "n_jobs":-1,
                  "objective":"binary:logistic", "eval_metric":"logloss"}
    if alg.lower()=="lightgbm" and LGBMClassifier is not None:
        params = {"n_estimators":trees, "learning_rate":0.1, "max_depth": (max_depth or -1),
                  "subsample":0.9, "colsample_bytree":0.9, "random_state":seed, "n_jobs":-1}
    model = init_model(alg, params)
    model, (Xte, yte, proba, preds, m) = train_model(model, df, test_size, seed)
    st.caption("Model trained on synthetic data.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("ROC AUC", f"{m['roc_auc']:.3f}")
    c3.metric("Test samples", f"{m['n_test']:,}")
    cm = np.array(m["confusion_matrix"])
    fig = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Pred", y="True"))
    st.plotly_chart(fig, use_container_width=True)
    rep_df = pd.DataFrame(m["report"]).T
    st.dataframe(rep_df, use_container_width=True)
with tabs[2]:
    st.subheader("Evaluate on Real Dataset (No Training)")
    up = st.file_uploader("Upload CSV or Parquet (≈1.1M rows OK)", type=["csv","parquet"])
    if up:
        fmt = "csv" if up.name.lower().endswith(".csv") else "parquet"
        st.info(f"Detected **{fmt.upper()}**.")
        sample = quick_profile_sample(up.getvalue(), fmt)
        st.write("Preview:")
        st.dataframe(sample.head(20), use_container_width=True)
        with st.spinner("Evaluating in chunks…"):
            res = evaluate_on_large_dataset(
                model, up.getvalue(), fmt, label_col=label_col,
                threshold=real_threshold, chunksize=chunk
            )
        if "error" in res:
            st.error(res["error"])
        else:
            met = res["metrics"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{met['n_rows']:,}")
            c2.metric("Accuracy", f"{met['accuracy']:.3f}")
            c3.metric("ROC AUC", f"{met['roc_auc']:.3f}")
            cm = np.array(met["confusion_matrix"])
            fig2 = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Pred", y="True"))
            st.plotly_chart(fig2, use_container_width=True)
            st.session_state["_phase2_cache"] = {
                "fmt": fmt, "file": up.getvalue(), "label_col": label_col,
                "threshold": real_threshold
            }
            st.session_state["_phase2_arrays"] = {
                "y": res["y"], "proba": res["proba"], "preds": res["preds"], "fail_mask": res["fail_mask"]
            }
            rep_df2 = pd.DataFrame(met["report"]).T
            st.dataframe(rep_df2, use_container_width=True)
    else:
        st.warning("Upload a dataset to run Phase 2.")
with tabs[3]:
    st.subheader("Multi-Agent Simulation (GA over decision policy)")
    if 'model' not in locals():
        st.error("Train a model in Phase 1 first.")
    else:
        with st.spinner("Simulating agents…"):
            sim = run_multi_agent(model, Xte, yte, n_agents=n_agents, generations=gens,
                                  seed=seed, base_threshold=base_t)
        st.success("Simulation complete.")
        st.write(f"**Best Agent** — threshold: `{sim['best_threshold']:.3f}`, "
                 f"risk: `{sim['best_risk']:.3f}`, accuracy: `{sim['best_acc']:.3f}`")
        hist = pd.DataFrame(sim["history"])
        st.dataframe(hist, use_container_width=True)
        fig3 = px.line(hist.melt(id_vars=["generation"], value_vars=["best_acc","mean_acc"]),
                       x="generation", y="value", color="variable",
                       title="Agent Fitness by Generation")
        st.plotly_chart(fig3, use_container_width=True)
        st.session_state["_phase3_best_t"] = sim["best_threshold"]
with tabs[4]:
    st.subheader("Failure-Driven Retraining")
    if 'model' not in locals():
        st.error("Train a model in Phase 1 first.")
    else:
        X_fail1, y_fail1 = gather_failures(Xte, yte, proba, preds, top_k=0)
        include_phase2 = st.checkbox("(Optional) Weight Phase 2 failure signal (no features) via threshold tuning",
                                     value=True)
        weight = st.slider("Failure weight (oversampling factor)", 1.0, 5.0, 2.0, step=0.5)
        do_retrain = st.button("Retrain on Failures")
        if do_retrain:
            with st.spinner("Retraining model using failure cases…"):
                X = df.drop(columns=["label"])
                y = df["label"].astype(int)
                Xtr, Xte2, ytr, yte2 = train_test_split(X, y, test_size=test_size,
                                                        random_state=seed, stratify=y)
                base_train_df = pd.concat([Xtr, ytr], axis=1)
                base_train_df.rename(columns={0:"label"}, inplace=True)
                fail_sets = [(X_fail1, y_fail1)]
                model_refined = failure_retrain(model, base_train_df, fail_sets,
                                                weight_failures=weight, seed=seed)
            st.success("Retrained.")
            proba2 = model_refined.predict_proba(Xte)[:,1]
            preds2 = (proba2 >= 0.5).astype(int)
            acc2 = accuracy_score(yte, preds2)
            auc2 = roc_auc_score(yte, proba2)
            c1, c2 = st.columns(2)
            c1.metric("Old Accuracy", f"{accuracy_score(yte, preds):.3f}")
            c2.metric("New Accuracy", f"{acc2:.3f}")
            c3, c4 = st.columns(2)
            c3.metric("Old ROC AUC", f"{roc_auc_score(yte, proba):.3f}")
            c4.metric("New ROC AUC", f"{auc2:.3f}")
            cm_old = confusion_matrix(yte, preds)
            cm_new = confusion_matrix(yte, preds2)
            fig_old = px.imshow(cm_old, text_auto=True, title="Before Retrain", aspect="auto")
            fig_new = px.imshow(cm_new, text_auto=True, title="After Retrain", aspect="auto")
            st.plotly_chart(fig_old, use_container_width=True)
            st.plotly_chart(fig_new, use_container_width=True)
            st.session_state["_refined_model"] = model_refined
with tabs[5]:
    st.subheader("Quick Prediction Pad (Gradio)")
    model_for_pad = st.session_state.get("_refined_model", None) or model
    if model_for_pad is None:
        st.warning("Train a model first.")
    else:
        feature_names = list(df.drop(columns=["label"]).columns)
        shared_q = queue.Queue()
        shared_q.put(model_for_pad)
        demo = build_gradio(shared_q, feature_names)
        port = find_free_port(7861)
        launch_gradio(demo, port)
        st.markdown(f"**Gradio running on port {port}** (embedded below):")
        st.components.v1.iframe(src=f"http://127.0.0.1:{port}", height=640)