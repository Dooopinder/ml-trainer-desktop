import os
import sys
import json
import gc
import traceback
from dataclasses import dataclass
from typing import Optional, Callable

import pandas as pd
import numpy as np

from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox,
    QComboBox, QTextEdit, QPlainTextEdit, QLineEdit,
    QTableWidget, QTableWidgetItem,
    QGroupBox, QDoubleSpinBox, QSpinBox, QSizePolicy,
    QSplitter
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from model_help import HELP_TEXT
from helpers import run_task, save_report_json, save_model_joblib


# -------------------------
# Config
# -------------------------
APP_TITLE = "ML Trainer (Desktop)"
TASKS = ["Classification", "Regression", "Clustering", "Anomaly Detection", "Dimensionality Reduction"]

MODELS = {
    "Classification": ["Logistic Regression", "Random Forest Classifier"],
    "Regression": ["Ridge Regression", "Random Forest Regressor"],
    "Clustering": ["KMeans"],
    "Anomaly Detection": ["Isolation Forest"],
    "Dimensionality Reduction": ["PCA"],
}


def app_dir() -> str:
    """Folder containing this script (works even if you move project)."""
    return os.path.dirname(os.path.abspath(__file__))


# -------------------------
# Matplotlib Canvas
# -------------------------
class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def clear(self):
        # prevent matplotlib figure buildup
        import matplotlib.pyplot as plt
        plt.close(self.fig)
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.draw()


# -------------------------
# AI Worker (background thread)
# -------------------------
class AiSignals(QObject):
    done = Signal(str)
    err = Signal(str)


@dataclass
class AiJob:
    provider: str            # "OpenAI (GPT)" or "Google (Gemini)"
    api_key: str
    model: str
    prompt: str


class AiWorker(QThread):
    def __init__(self, job: AiJob, signals: AiSignals):
        super().__init__()
        self.job = job
        self.signals = signals

    def run(self):
        try:
            text = self._call_provider(self.job)
            self.signals.done.emit(text)
        except Exception as e:
            self.signals.err.emit(f"{e}\n\n{traceback.format_exc()}")

    @staticmethod
    def _call_provider(job: AiJob) -> str:
        provider = job.provider.strip()
        api_key = job.api_key.strip()
        model = job.model.strip()
        prompt = job.prompt

        if provider == "OpenAI (GPT)":
            # pip install openai
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            # Responses API (recommended)
            resp = client.responses.create(
                model=model,
                input=prompt
            )
            return (resp.output_text or "").strip() or "[No text returned]"

        if provider == "Google (Gemini)":
            # pip install google-genai
            from google import genai
            client = genai.Client(api_key=api_key)

            # IMPORTANT: model should be like "gemini-2.5-flash" (not "models/...")
            resp = client.models.generate_content(
                model=model,
                contents=prompt
            )
            text = getattr(resp, "text", None)
            return (text or "").strip() or "[No text returned]"

        return "Select a provider (OpenAI or Gemini)."


# -------------------------
# Main Window (no .ui file)
# -------------------------
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setAttribute(Qt.WA_DeleteOnClose)

        # State
        self.df: Optional[pd.DataFrame] = None
        self.artifact = None
        self.results_df: Optional[pd.DataFrame] = None
        self.report: Optional[dict] = None
        self.plot_payload: Optional[dict] = None
        self.loaded_csv_path: Optional[str] = None

        self.ai_thread: Optional[AiWorker] = None
        self.ai_signals = AiSignals()

        self._build_ui()
        self._wire_signals()
        self._init_defaults()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        # Split left/right
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # LEFT PANEL
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)

        grp_controls = QGroupBox("Controls")
        controls = QVBoxLayout(grp_controls)

        self.btn_load = QPushButton("Load CSV")
        controls.addWidget(self.btn_load)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        controls.addWidget(self.lbl_file)

        controls.addWidget(QLabel("Task"))
        self.cmb_task = QComboBox()
        controls.addWidget(self.cmb_task)

        controls.addWidget(QLabel("Model"))
        self.cmb_model = QComboBox()
        controls.addWidget(self.cmb_model)

        self.lbl_model_usage = QLabel("")
        self.lbl_model_usage.setWordWrap(True)
        self.lbl_model_usage.setStyleSheet("color: #444;")
        controls.addWidget(self.lbl_model_usage)

        controls.addWidget(QLabel("Target column (Classification/Regression only)"))
        self.cmb_target = QComboBox()
        controls.addWidget(self.cmb_target)

        # test size / random state
        grid = QGridLayout()
        grid.addWidget(QLabel("Test size"), 0, 0)
        self.spin_test_size = QDoubleSpinBox()
        self.spin_test_size.setMinimum(0.05)
        self.spin_test_size.setMaximum(0.95)
        self.spin_test_size.setSingleStep(0.05)
        self.spin_test_size.setDecimals(2)
        grid.addWidget(self.spin_test_size, 0, 1)

        grid.addWidget(QLabel("Random state"), 1, 0)
        self.spin_random_state = QSpinBox()
        self.spin_random_state.setMaximum(999999)
        grid.addWidget(self.spin_random_state, 1, 1)
        controls.addLayout(grid)

        row = QHBoxLayout()
        self.btn_help = QPushButton("Help")
        self.btn_run = QPushButton("Run")
        row.addWidget(self.btn_help)
        row.addWidget(self.btn_run)
        controls.addLayout(row)

        self.btn_export_csv = QPushButton("Export Results CSV")
        self.btn_export_model = QPushButton("Export Model (.joblib)")
        self.btn_export_report = QPushButton("Export Report (.json)")
        controls.addWidget(self.btn_export_csv)
        controls.addWidget(self.btn_export_model)
        controls.addWidget(self.btn_export_report)

        controls.addStretch(1)
        left_layout.addWidget(grp_controls)

        # AI PANEL
        grp_ai = QGroupBox("AI Assistant (Live)")
        ai = QVBoxLayout(grp_ai)

        r0 = QHBoxLayout()
        r0.addWidget(QLabel("Provider"))
        self.cmb_ai_provider = QComboBox()
        self.cmb_ai_provider.addItems(["None", "OpenAI (GPT)", "Google (Gemini)"])
        r0.addWidget(self.cmb_ai_provider)
        ai.addLayout(r0)

        self.txt_ai_key = QLineEdit()
        self.txt_ai_key.setEchoMode(QLineEdit.Password)
        self.txt_ai_key.setPlaceholderText("API Key")
        ai.addWidget(self.txt_ai_key)

        self.txt_ai_model = QLineEdit()
        self.txt_ai_model.setPlaceholderText("Model (OpenAI: gpt-4.1-mini / Gemini: gemini-2.5-flash)")
        ai.addWidget(self.txt_ai_model)

        self.txt_ai_prompt = QPlainTextEdit()
        self.txt_ai_prompt.setPlaceholderText("Ask: explain results, suggest next model, data issues, next steps...")
        self.txt_ai_prompt.setMinimumHeight(110)
        ai.addWidget(self.txt_ai_prompt)

        r1 = QHBoxLayout()
        self.btn_ai_fill = QPushButton("Fill prompt from results")
        self.btn_ai_ask = QPushButton("Ask AI (Live)")
        r1.addWidget(self.btn_ai_fill)
        r1.addWidget(self.btn_ai_ask)
        ai.addLayout(r1)

        self.txt_ai_response = QTextEdit()
        self.txt_ai_response.setReadOnly(True)
        self.txt_ai_response.setMinimumHeight(150)
        ai.addWidget(self.txt_ai_response)

        left_layout.addWidget(grp_ai)

        # RIGHT PANEL
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)

        right_layout.addWidget(QLabel("Preview"))
        self.tbl_preview = QTableWidget()
        self.tbl_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.tbl_preview, 3)

        right_layout.addWidget(QLabel("Results"))
        self.txt_results = QTextEdit()
        self.txt_results.setReadOnly(True)
        self.txt_results.setStyleSheet("font-size: 13px;")
        right_layout.addWidget(self.txt_results, 1)

        right_layout.addWidget(QLabel("Plot"))
        self.canvas = PlotCanvas()
        right_layout.addWidget(self.canvas, 3)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(root)
        layout.addWidget(splitter)

    def _wire_signals(self):
        self.btn_load.clicked.connect(self.load_csv)
        self.btn_help.clicked.connect(self.show_help)
        self.btn_run.clicked.connect(self.run)

        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_model.clicked.connect(self.export_model)
        self.btn_export_report.clicked.connect(self.export_report)

        self.cmb_task.currentTextChanged.connect(self.on_task_changed)
        self.cmb_model.currentTextChanged.connect(self.on_model_changed)

        self.btn_ai_fill.clicked.connect(self.ai_fill_prompt)
        self.btn_ai_ask.clicked.connect(self.ai_ask_live)

        self.ai_signals.done.connect(self._ai_done)
        self.ai_signals.err.connect(self._ai_err)

    def _init_defaults(self):
        self.cmb_task.clear()
        self.cmb_task.addItems(TASKS)

        self.spin_test_size.setValue(0.20)
        self.spin_random_state.setValue(42)

        self.on_task_changed(self.cmb_task.currentText())

        self.txt_results.setText("Load a CSV to begin.")
        self.canvas.clear()

        # set smart defaults for models
        self._apply_ai_defaults()

    # -------------------------
    # Controls
    # -------------------------
    def on_task_changed(self, task_name: str):
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        self.cmb_model.addItems(MODELS[task_name])
        self.cmb_model.blockSignals(False)

        supervised = task_name in ("Classification", "Regression")
        self.cmb_target.setEnabled(supervised)
        self.spin_test_size.setEnabled(supervised)
        self.spin_random_state.setEnabled(supervised)

        self.on_model_changed(self.cmb_model.currentText())

    def on_model_changed(self, _=None):
        task = self.cmb_task.currentText()
        model = self.cmb_model.currentText()
        self.lbl_model_usage.setText(HELP_TEXT.get(task, {}).get(model, ""))

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", app_dir(), "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read CSV:\n{e}")
            return

        self.df = df
        self.loaded_csv_path = path

        self.lbl_file.setText(path)
        self.cmb_target.clear()
        self.cmb_target.addItems(self.df.columns.tolist())

        self.fill_preview(self.df.head(30))
        self.txt_results.setText("Loaded dataset. Choose task/model and click Run.")
        self.canvas.clear()
        gc.collect()

    def fill_preview(self, df: pd.DataFrame):
        self.tbl_preview.setRowCount(len(df))
        self.tbl_preview.setColumnCount(len(df.columns))
        self.tbl_preview.setHorizontalHeaderLabels([str(c) for c in df.columns.tolist()])
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                self.tbl_preview.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
        self.tbl_preview.resizeColumnsToContents()

    def show_help(self):
        task = self.cmb_task.currentText()
        lines = [f"{task} models:\n"]
        for m in MODELS[task]:
            lines.append(f"- {m}: {HELP_TEXT.get(task, {}).get(m, '')}")
        QMessageBox.information(self, "Help", "\n".join(lines))

    # -------------------------
    # Run ML
    # -------------------------
    def run(self):
        if self.df is None:
            QMessageBox.warning(self, "Missing data", "Load a CSV first.")
            return

        self.btn_run.setEnabled(False)
        QApplication.processEvents()

        task = self.cmb_task.currentText()
        model = self.cmb_model.currentText()
        target = self.cmb_target.currentText() if self.cmb_target.isEnabled() else None

        try:
            self.artifact, self.results_df, self.report, self.plot_payload = run_task(
                self.df,
                task,
                model,
                target,
                test_size=float(self.spin_test_size.value()),
                random_state=int(self.spin_random_state.value()),
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.btn_run.setEnabled(True)
            return

        self.render_results()
        self.render_plot()

        gc.collect()
        self.btn_run.setEnabled(True)

    def render_results(self):
        if not self.report:
            return

        task = self.report.get("task")
        model = self.report.get("model")
        lines = [f"Task: {task}", f"Model: {model}", ""]

        if task == "Classification":
            lines += [
                "Metrics",
                f"  • Accuracy: {self.report.get('accuracy'):.3f}",
                f"  • F1 (weighted): {self.report.get('f1_weighted'):.3f}",
            ]
        elif task == "Regression":
            lines += [
                "Metrics",
                f"  • RMSE: {self.report.get('rmse'):.3f}",
                f"  • R²: {self.report.get('r2'):.3f}",
            ]
        elif task == "Clustering":
            lines.append("Cluster counts")
            cc = self.report.get("cluster_counts", {})
            for k, v in cc.items():
                lines.append(f"  • Cluster {k}: {v} rows")
        elif task == "Anomaly Detection":
            lines.append(f"Anomalies found: {self.report.get('anomalies')}")
        elif task == "Dimensionality Reduction":
            evr = self.report.get("explained_variance_ratio")
            if evr:
                lines.append("Explained variance")
                if len(evr) > 0:
                    lines.append(f"  • PC1: {evr[0]:.3f}")
                if len(evr) > 1:
                    lines.append(f"  • PC2: {evr[1]:.3f}")

        self.txt_results.setText("\n".join(lines))

    def render_plot(self):
        self.canvas.clear()
        p = self.plot_payload or {}
        ptype = p.get("type")
        ax = self.canvas.ax

        if ptype == "confusion_matrix":
            cm = p["cm"]
            labels = p["labels"]
            ax.imshow(cm)
            ax.set_aspect("equal")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11)

        elif ptype == "pred_vs_actual":
            y_true = np.array(p["y_true"])
            y_pred = np.array(p["y_pred"])
            ax.scatter(y_true, y_pred, alpha=0.6)
            ax.set_title("Predicted vs Actual")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            mn = float(np.nanmin([np.min(y_true), np.min(y_pred)]))
            mx = float(np.nanmax([np.max(y_true), np.max(y_pred)]))
            ax.plot([mn, mx], [mn, mx])

        elif ptype == "scatter2d":
            x = p["x"]
            y = p["y"]
            c = p.get("c")
            ax.set_title(p.get("title", "Scatter"))
            if c is None:
                ax.scatter(x, y, alpha=0.6)
            else:
                ax.scatter(x, y, c=c, alpha=0.7)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        elif ptype == "hist":
            vals = p["values"]
            ax.hist(vals, bins=30)
            ax.set_title(p.get("title", "Histogram"))
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")

        else:
            ax.text(0.5, 0.5, "No plot for this result.", ha="center", va="center")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    # -------------------------
    # Export
    # -------------------------
    def export_csv(self):
        if self.results_df is None:
            QMessageBox.warning(self, "Nothing to export", "Run first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "results.csv", "CSV Files (*.csv)")
        if path:
            self.results_df.to_csv(path, index=False)

    def export_model(self):
        if self.artifact is None:
            QMessageBox.warning(self, "Nothing to export", "Run first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "model.joblib", "Joblib (*.joblib)")
        if path:
            save_model_joblib(path, self.artifact)

    def export_report(self):
        if self.report is None:
            QMessageBox.warning(self, "Nothing to export", "Run first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Report", "report.json", "JSON (*.json)")
        if path:
            save_report_json(path, self.report)

    # -------------------------
    # AI Assistant (Live)
    # -------------------------
    def _apply_ai_defaults(self):
        provider = self.cmb_ai_provider.currentText()

        # Do not override if user already typed something
        cur = self.txt_ai_model.text().strip()

        if provider == "Google (Gemini)" and not cur:
            # Fix your 404: use correct modern model IDs (NOT "models/...")
            self.txt_ai_model.setText("gemini-2.5-flash")
        elif provider == "OpenAI (GPT)" and not cur:
            # Pick something common; user can change to what their key supports
            self.txt_ai_model.setText("gpt-4.1-mini")

    def ai_fill_prompt(self):
        if self.df is None:
            QMessageBox.information(self, "AI", "Load a CSV first so I can include your dataset.")
            return

        # Can work even before running ML
        report = self.report or {}
        prompt = self.build_ai_context_prompt(
            user_question="Explain what this dataset looks like and what I should try next.",
            df=self.df,
            report=report
        )
        self.txt_ai_prompt.setPlainText(prompt)

    def build_ai_context_prompt(self, user_question: str, df: pd.DataFrame, report: dict) -> str:
        # Keep it small (cheap + fast + avoids token blow-up)
        preview_rows = min(12, len(df))
        head_csv = df.head(preview_rows).to_csv(index=False)

        cols = list(df.columns)
        dtypes = {c: str(df[c].dtype) for c in cols}
        missing = df.isna().sum().sort_values(ascending=False).head(10).to_dict()

        # compact report
        safe_report = report if report else {}
        try:
            report_json = json.dumps(safe_report, indent=2, default=str)
        except Exception:
            report_json = str(safe_report)

        csv_name = os.path.basename(self.loaded_csv_path) if self.loaded_csv_path else "(not saved)"

        return f"""
You are an ML assistant for non-technical users.
Be clear. Be practical. No jargon.

USER QUESTION:
{user_question}

CSV FILE:
{csv_name}

MODEL REPORT (metrics + settings):
{report_json}

DATASET SHAPE:
rows={len(df)}, cols={len(df.columns)}

COLUMNS + DTYPES:
{json.dumps(dtypes, indent=2)}

TOP MISSING COUNTS (top 10):
{json.dumps(missing, indent=2)}

DATA PREVIEW (first {preview_rows} rows as CSV):
{head_csv}

Answer format:
- Short summary (2-3 lines)
- What the dataset contains
- If you see issues (missing values, wrong target type, leakage risk)
- 3 next steps
""".strip()

    def ai_ask_live(self):
        provider = self.cmb_ai_provider.currentText().strip()
        api_key = self.txt_ai_key.text().strip()
        model = self.txt_ai_model.text().strip()
        user_question = self.txt_ai_prompt.toPlainText().strip()

        if provider == "None":
            self.txt_ai_response.setText("Pick OpenAI (GPT) or Google (Gemini).")
            return

        if not api_key:
            self.txt_ai_response.setText("API key missing. Paste your key to enable live mode.")
            return

        if not model:
            self._apply_ai_defaults()
            model = self.txt_ai_model.text().strip()

        if not user_question:
            QMessageBox.information(self, "AI", "Type a question (or click 'Fill prompt from results').")
            return

        if self.df is None:
            QMessageBox.information(self, "AI", "Load a CSV first so the AI can read the dataset.")
            return

        # Build context prompt (includes CSV + report)
        prompt = self.build_ai_context_prompt(
            user_question=user_question,
            df=self.df,
            report=(self.report or {})
        )

        # Fire background thread
        self.btn_ai_ask.setEnabled(False)
        self.txt_ai_response.setText("Thinking... (live API call)")

        job = AiJob(provider=provider, api_key=api_key, model=model, prompt=prompt)
        self.ai_thread = AiWorker(job, self.ai_signals)
        self.ai_thread.start()

    def _ai_done(self, text: str):
        self.txt_ai_response.setText(text)
        self.btn_ai_ask.setEnabled(True)

    def _ai_err(self, err: str):
        # Friendly tip for your Gemini 404 case
        tip = ""
        if "models/" in err or "NOT_FOUND" in err or "is not found" in err:
            tip = (
                "\n\nTip:\n"
                "- For Gemini, set model to: gemini-2.5-flash (or gemini-2.0-flash-001)\n"
                "- Do NOT type: models/gemini-...\n"
            )
        self.txt_ai_response.setText("AI ERROR:\n" + err + tip)
        self.btn_ai_ask.setEnabled(True)

    # -------------------------
    # Shutdown
    # -------------------------
    def closeEvent(self, event):
        try:
            self.canvas.clear()
        except Exception:
            pass
        self.df = None
        self.results_df = None
        self.artifact = None
        self.report = None
        self.plot_payload = None
        gc.collect()
        event.accept()


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    os.chdir(app_dir())  # dynamic: always run relative to script folder
    app = QApplication(sys.argv)
    w = Main()
    w.resize(1250, 800)
    w.show()
    sys.exit(app.exec())
