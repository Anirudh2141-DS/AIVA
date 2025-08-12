# AIVA Dashboard – Streamlit + Gradio

## Overview
AIVA (**AI-Driven Multi-Agent Venture Capital Simulation Pipeline**) is an interactive dashboard that blends **Streamlit** and **Gradio** to simulate venture capital decision-making using machine learning, genetic algorithms, and the PED framework (Predict → Evaluate → Decide).

This dashboard provides:
- **Phase 1**: Synthetic data generation and training of an initial predictive model.
- **Phase 3** (demo mode): Multi-agent simulation with threshold evolution via a toy GA.
- **Live prediction panel**: Score JSON-formatted startup features instantly via Gradio.
- Metrics tracking, data snapshots, and visualizations.

> ⚠️ The current simulation loop in this dashboard uses a simplified GA placeholder.  
> Integrating the full PED + GA logic from the core `AIVA.ipynb` will enable Phases 2–4 from the [system overview](overview.pdf).

---

## Features
- **Synthetic Data Controls**: Adjust rows, features, class separability from the sidebar.
- **RandomForest Model Training**: Train/retrain with adjustable hyperparameters.
- **Metrics Tab**: View accuracy, ROC AUC, confusion matrix, and classification report.
- **Agent Simulation Tab**: Run a multi-agent GA evolution loop, visualize accuracy trends.
- **Gradio Prediction Tab**: Paste JSON startup features and get probability & decision instantly.
- **Hybrid UI**: Streamlit for control + telemetry, Gradio for quick scoring.

---

## Installation

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd <your-repo-folder>
