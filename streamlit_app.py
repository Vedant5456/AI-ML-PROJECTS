"""
E-Signature Fraud Detection System
Run: streamlit run streamlit_app.py
"""
import os, sys, json, time, pickle
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from streamlit_drawable_canvas import st_canvas

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)

from utils.feature_extractor import extract_features, build_dataset, build_cnn_dataset
from models.classifiers import (train_svm, train_random_forest,
                                 train_knn, train_logistic_regression, train_cnn)

st.set_page_config(
    page_title="E-Signature Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #020818; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #0a1628; border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
#MainMenu, footer, header { visibility: hidden; }
.stButton > button {
    background: linear-gradient(135deg, #1a56db, #0ea5e9);
    color: white; font-weight: 700; border: none;
    border-radius: 8px; width: 100%;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
.stTabs [data-baseweb="tab-list"] {
    background: #0a1628; border-radius: 10px; padding: 5px; gap: 5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #64748b;
    font-weight: 600; font-size: 1rem; padding: 0.6rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#1a56db,#0ea5e9); color: white !important;
}
.stProgress > div > div { background: linear-gradient(90deg, #1a56db, #0ea5e9); }
[data-testid="stMetric"] {
    background: #0a1628; border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 0.75rem 1rem;
}
[data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: 800; }
hr { border-color: #1e3a5f; }
.stSelectbox > div > div { background: #0d1f3c; border: 1px solid #1e3a5f; }
</style>
""", unsafe_allow_html=True)


def load_results():
    path = os.path.join(REPORTS_DIR, "training_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def models_trained():
    trained = []
    for m in ["svm", "random_forest", "knn", "logistic_regression"]:
        if os.path.exists(os.path.join(MODELS_DIR, f"{m}.pkl")):
            trained.append(m)
    if os.path.exists(os.path.join(MODELS_DIR, "cnn_model.h5")):
        trained.append("cnn")
    return trained


def load_clf(name):
    with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


def run_inference(img, model_name):
    tmp = os.path.join(BASE_DIR, "_tmp_inf.png")
    img.convert("L").save(tmp)
    try:
        if model_name == "cnn":
            import tensorflow as tf
            from utils.feature_extractor import load_and_preprocess
            cnn_path = os.path.join(MODELS_DIR, "cnn_model.h5")
            if not os.path.exists(cnn_path):
                raise FileNotFoundError("CNN model not found. Run: python train_cnn.py")
            model = tf.keras.models.load_model(cnn_path)
            arr = load_and_preprocess(tmp, for_cnn=True)
            arr = arr[np.newaxis, ..., np.newaxis]
            prob = float(model.predict(arr, verbose=0)[0][0])
        else:
            pkl_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"{model_name} not found. Run: python train.py")
            clf = load_clf(model_name)
            feat = extract_features(tmp).reshape(1, -1)
            prob = float(clf.predict_proba(feat)[0][1])
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return prob


def verdict_from_prob(prob):
    if prob < 0.35:
        return "GENUINE", "#22c55e", "✔"
    elif prob < 0.65:
        return "SUSPICIOUS", "#eab308", "⚠"
    else:
        return "FORGED", "#ef4444", "✗"


MODEL_LABELS = {
    "svm": "SVM",
    "random_forest": "Random Forest",
    "knn": "KNN",
    "logistic_regression": "Logistic Regression",
    "cnn": "CNN",
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ══════════════════════════════════════════════
# LOGIN PAGE640

# ══════════════════════════════════════════════
if not st.session_state.logged_in:
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; margin-bottom:2rem;'>
            <div style='font-size:4rem;'>🏦</div>
            <div style='font-size:1.8rem; font-weight:800; color:#e2e8f0; margin-top:0.5rem;'>
                E-Signature Fraud Detection
            </div>
            <div style='font-size:0.82rem; color:#475569; margin-top:0.4rem;
                        letter-spacing:3px; text-transform:uppercase;'>
                Secure Verification System
            </div>
        </div>
        <div style='background:#0a1628; border:1px solid #1e3a5f;
                    border-radius:16px; padding:2.5rem;'>
            <div style='font-size:0.8rem; font-weight:700; color:#475569;
                        margin-bottom:1.5rem; text-align:center;
                        letter-spacing:3px; text-transform:uppercase;'>
                Officer Login
            </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if st.button("🔐 Login", use_container_width=True):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

        st.markdown("""
        </div>
        <div style='text-align:center; margin-top:1rem; font-size:0.75rem; color:#1e3a5f;'>
            Username: admin &nbsp;|&nbsp; Password: admin123
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div style='background:#0a1628; border:1px solid #1e3a5f; border-radius:12px;
            padding:1rem 2rem; margin-bottom:1.5rem;
            display:flex; justify-content:space-between; align-items:center;'>
    <div style='display:flex; align-items:center; gap:1rem;'>
        <div style='font-size:1.8rem;'>🏦</div>
        <div>
            <div style='font-size:1.1rem; font-weight:800; color:#e2e8f0;'>
                E-Signature Fraud Detection System
            </div>
            <div style='font-size:0.72rem; color:#475569;
                        letter-spacing:2px; text-transform:uppercase;'>
                Secure · Reliable · AI Powered
            </div>
        </div>
    </div>
    <div style='font-size:0.75rem; color:#475569; font-family:monospace;'>
        🟢 System Online &nbsp;|&nbsp; Officer: Admin
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════
trained = models_trained()
results = load_results()
model_results = results.get("results", [])

tab1, tab2, tab3 = st.tabs([
    "✍  Verify Signature",
    "🔍  Compare Signatures",
    "⚙  System & Training",
])


# ══════════════════════════════════════════════
# TAB 1 – VERIFY SIGNATURE
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### ✍ Verify Signature")
    st.markdown("Draw or upload a signature to instantly verify its authenticity.")
    st.markdown("---")

    if not trained:
        st.warning("⚠ No models trained. Go to **⚙ System & Training** tab first.")
    else:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("#### Input")
            input_mode = st.radio(
                "Method",
                ["✏ Draw Signature", "📁 Upload Image"],
                horizontal=True,
                label_visibility="collapsed",
                key="v_mode",
            )

            sig_image = None

            if input_mode == "✏ Draw Signature":
                st.markdown("**Draw below:**")
                canvas = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=3,
                    stroke_color="#000000",
                    background_color="#ffffff",
                    height=220,
                    width=480,
                    drawing_mode="freedraw",
                    key="v_canvas",
                )
                if canvas.image_data is not None:
                    arr = canvas.image_data
                    if arr[:, :, 3].sum() > 1000:
                        sig_image = Image.fromarray(arr.astype("uint8"), "RGBA")
            else:
                up = st.file_uploader(
                    "Upload",
                    type=["jpg", "jpeg", "png", "bmp"],
                    label_visibility="collapsed",
                    key="v_upload",
                )
                if up:
                    sig_image = Image.open(up).convert("RGBA")
                    st.image(sig_image, caption="Uploaded", use_column_width=True)

            st.markdown("#### Select Model")
            mopts = {MODEL_LABELS[m]: m for m in trained}
            slabel = st.selectbox(
                "Model", list(mopts.keys()),
                label_visibility="collapsed",
                key="v_model",
            )
            smodel = mopts[slabel]

            st.markdown(f"""
            <div style='background:#0d1f3c; border:1px solid #1e3a5f;
                        border-radius:8px; padding:0.75rem 1rem;
                        font-family:monospace; font-size:0.8rem; margin-top:0.5rem;'>
                Model: <span style='color:#60a5fa;'>{slabel}</span>
                &nbsp;·&nbsp;
                Status: <span style='color:#22c55e;'>✔ Ready</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            vbtn = st.button("🔍 Verify Signature", use_container_width=True, key="v_btn")

        with col_right:
            st.markdown("#### Result")
            if vbtn:
                if sig_image is None:
                    st.warning("⚠ Please draw or upload a signature first.")
                else:
                    with st.spinner("Analysing signature..."):
                        try:
                            prob = run_inference(sig_image, smodel)
                            verdict, color, icon = verdict_from_prob(prob)
                            gp = round((1 - prob) * 100, 1)
                            fp = round(prob * 100, 1)

                            if verdict == "GENUINE":
                                bg = "rgba(34,197,94,0.08)"
                            elif verdict == "FORGED":
                                bg = "rgba(239,68,68,0.08)"
                            else:
                                bg = "rgba(234,179,8,0.08)"

                            st.markdown(f"""
                            <div style='background:{bg}; border:2px solid {color};
                                        border-radius:14px; padding:2rem;
                                        text-align:center; margin-bottom:1.5rem;'>
                                <div style='font-size:3.5rem;'>{icon}</div>
                                <div style='font-size:2.2rem; font-weight:800;
                                            color:{color}; letter-spacing:4px;'>
                                    {verdict}
                                </div>
                                <div style='color:#64748b; font-size:0.82rem; margin-top:0.5rem;'>
                                    Model: {slabel} &nbsp;·&nbsp; {time.strftime("%H:%M:%S")}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            c1, c2 = st.columns(2)
                            c1.metric("Genuine Score", f"{gp}%")
                            c2.metric("Forgery Score", f"{fp}%")

                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=fp,
                                title={"text": "Forgery Risk",
                                       "font": {"color": "#e2e8f0", "size": 13}},
                                number={"suffix": "%",
                                        "font": {"color": color, "size": 34}},
                                gauge={
                                    "axis": {"range": [0, 100],
                                             "tickcolor": "#475569"},
                                    "bar": {"color": color},
                                    "bgcolor": "#0a1628",
                                    "steps": [
                                        {"range": [0, 35],
                                         "color": "rgba(34,197,94,0.1)"},
                                        {"range": [35, 65],
                                         "color": "rgba(234,179,8,0.1)"},
                                        {"range": [65, 100],
                                         "color": "rgba(239,68,68,0.1)"},
                                    ],
                                },
                            ))
                            fig.update_layout(
                                paper_bgcolor="#0a1628",
                                font_color="#e2e8f0",
                                height=250,
                                margin=dict(t=50, b=10, l=20, r=20),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ {e}")
            else:
                st.markdown("""
                <div style='text-align:center; padding:5rem 1rem;
                            border:1px dashed #1e3a5f; border-radius:12px;'>
                    <div style='font-size:3rem; opacity:0.15;'>🔍</div>
                    <div style='color:#1e3a5f; margin-top:0.75rem; font-size:0.9rem;'>
                        Draw or upload a signature<br/>then click Verify
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 – COMPARE SIGNATURES
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 🔍 Compare Signatures")
    st.markdown("Compare a reference signature against a test signature side by side.")
    st.markdown("---")

    if not trained:
        st.warning("⚠ No models trained. Go to **⚙ System & Training** tab first.")
    else:
        cm_opts = {MODEL_LABELS[m]: m for m in trained}
        c_ms, _ = st.columns([2, 4])
        with c_ms:
            cm_label = st.selectbox("Model", list(cm_opts.keys()), key="cm_model")
            cm_model = cm_opts[cm_label]

        st.markdown("---")
        col_ref, col_test = st.columns(2, gap="large")
        ref_image = None
        test_image = None

        with col_ref:
            st.markdown("""
            <div style='background:#0a1628; border:1px solid #1e3a5f;
                        border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.75rem;'>
                <span style='font-size:0.72rem; color:#60a5fa;
                             font-family:monospace; letter-spacing:1px; font-weight:700;'>
                    📋 REFERENCE SIGNATURE
                </span><br/>
                <span style='font-size:0.78rem; color:#64748b;'>Known genuine signature</span>
            </div>
            """, unsafe_allow_html=True)

            ref_mode = st.radio(
                "Ref", ["✏ Draw", "📁 Upload"],
                horizontal=True, key="ref_mode",
                label_visibility="collapsed",
            )
            if ref_mode == "✏ Draw":
                rc = st_canvas(
                    stroke_width=3, stroke_color="#000000",
                    background_color="#ffffff", height=180, width=430,
                    drawing_mode="freedraw", key="ref_canvas",
                )
                if rc.image_data is not None:
                    arr = rc.image_data
                    if arr[:, :, 3].sum() > 1000:
                        ref_image = Image.fromarray(arr.astype("uint8"), "RGBA")
            else:
                ru = st.file_uploader(
                    "Ref upload", type=["jpg", "jpeg", "png"],
                    key="ref_up", label_visibility="collapsed",
                )
                if ru:
                    ref_image = Image.open(ru).convert("RGBA")
                    st.image(ref_image, use_column_width=True)

        with col_test:
            st.markdown("""
            <div style='background:#0a1628; border:1px solid #1e3a5f;
                        border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.75rem;'>
                <span style='font-size:0.72rem; color:#8b5cf6;
                             font-family:monospace; letter-spacing:1px; font-weight:700;'>
                    🔎 TEST SIGNATURE
                </span><br/>
                <span style='font-size:0.78rem; color:#64748b;'>Signature to verify</span>
            </div>
            """, unsafe_allow_html=True)

            test_mode = st.radio(
                "Test", ["✏ Draw", "📁 Upload"],
                horizontal=True, key="test_mode",
                label_visibility="collapsed",
            )
            if test_mode == "✏ Draw":
                tc = st_canvas(
                    stroke_width=3, stroke_color="#000000",
                    background_color="#ffffff", height=180, width=430,
                    drawing_mode="freedraw", key="test_canvas",
                )
                if tc.image_data is not None:
                    arr = tc.image_data
                    if arr[:, :, 3].sum() > 1000:
                        test_image = Image.fromarray(arr.astype("uint8"), "RGBA")
            else:
                tu = st.file_uploader(
                    "Test upload", type=["jpg", "jpeg", "png"],
                    key="test_up", label_visibility="collapsed",
                )
                if tu:
                    test_image = Image.open(tu).convert("RGBA")
                    st.image(test_image, use_column_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        cbtn = st.button(
            "🔍 Compare Signatures", use_container_width=True, key="cm_btn"
        )

        if cbtn:
            if ref_image is None or test_image is None:
                st.warning("⚠ Please provide both Reference and Test signatures.")
            else:
                with st.spinner("Comparing..."):
                    try:
                        rp = run_inference(ref_image, cm_model)
                        tp = run_inference(test_image, cm_model)
                        sim = round(100 - abs(rp - tp) * 100, 1)

                        if sim >= 75:
                            mv, mc, mi = "MATCH", "#22c55e", "✔"
                        elif sim >= 50:
                            mv, mc, mi = "UNCERTAIN", "#eab308", "⚠"
                        else:
                            mv, mc, mi = "NO MATCH", "#ef4444", "✗"

                        st.markdown("---")

                        if mv == "MATCH":
                            bg = "rgba(34,197,94,0.08)"
                        elif mv == "NO MATCH":
                            bg = "rgba(239,68,68,0.08)"
                        else:
                            bg = "rgba(234,179,8,0.08)"

                        st.markdown(f"""
                        <div style='background:{bg}; border:2px solid {mc};
                                    border-radius:14px; padding:2rem;
                                    text-align:center; margin-bottom:1.5rem;'>
                            <div style='font-size:3.5rem;'>{mi}</div>
                            <div style='font-size:2.2rem; font-weight:800;
                                        color:{mc}; letter-spacing:4px;'>{mv}</div>
                            <div style='font-size:1rem; color:#94a3b8; margin-top:0.5rem;'>
                                Similarity: <strong style='color:{mc};'>{sim}%</strong>
                                &nbsp;·&nbsp; Model: {cm_label}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Reference Score", f"{round((1 - rp) * 100, 1)}%")
                        c2.metric("Test Score", f"{round((1 - tp) * 100, 1)}%")
                        c3.metric("Similarity", f"{sim}%")

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=sim,
                            title={"text": "Signature Similarity",
                                   "font": {"color": "#e2e8f0", "size": 13}},
                            number={"suffix": "%",
                                    "font": {"color": mc, "size": 34}},
                            delta={
                                "reference": 75,
                                "increasing": {"color": "#22c55e"},
                                "decreasing": {"color": "#ef4444"},
                            },
                            gauge={
                                "axis": {"range": [0, 100],
                                         "tickcolor": "#475569"},
                                "bar": {"color": mc},
                                "bgcolor": "#0a1628",
                                "steps": [
                                    {"range": [0, 50],
                                     "color": "rgba(239,68,68,0.1)"},
                                    {"range": [50, 75],
                                     "color": "rgba(234,179,8,0.1)"},
                                    {"range": [75, 100],
                                     "color": "rgba(34,197,94,0.1)"},
                                ],
                                "threshold": {
                                    "line": {"color": "white", "width": 3},
                                    "value": 75,
                                },
                            },
                        ))
                        fig.update_layout(
                            paper_bgcolor="#0a1628",
                            font_color="#e2e8f0",
                            height=280,
                            margin=dict(t=50, b=10, l=20, r=20),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ {e}")


# ══════════════════════════════════════════════
# TAB 3 – SYSTEM & TRAINING
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### ⚙ System & Training")
    st.markdown("---")

    sys_tab1, sys_tab2 = st.tabs(["📊 Dashboard", "🔧 Train Models"])

    with sys_tab1:
        dataset_info = results.get("dataset_info", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples", dataset_info.get("total", "—"))
        c2.metric("Genuine Signatures", dataset_info.get("genuine", "—"))
        c3.metric("Forged Signatures", dataset_info.get("forged", "—"))
        c4.metric("Models Trained", f"{len(trained)}/5")

        if model_results:
            best = max(model_results, key=lambda r: r["accuracy"])
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(26,86,219,0.1),
                        rgba(14,165,233,0.08)); border:1px solid #1a56db;
                        border-radius:14px; padding:1.5rem 2rem; margin:1rem 0;
                        display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <div style='font-size:0.7rem; color:#60a5fa;
                                font-family:monospace; letter-spacing:2px;'>
                        🏆 BEST MODEL
                    </div>
                    <div style='font-size:1.4rem; font-weight:800;
                                color:#e2e8f0; margin:0.3rem 0;'>
                        {best["model"]}
                    </div>
                    <div style='color:#64748b; font-size:0.82rem;'>
                        Accuracy {best["accuracy"]}%
                        · F1 {best["f1"]}%
                        · Precision {best["precision"]}%
                    </div>
                </div>
                <div style='font-size:2.8rem; font-weight:800;
                            color:#60a5fa; font-family:monospace;'>
                    {best["accuracy"]}<span style='font-size:1.2rem;'>%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            df = pd.DataFrame(model_results)
            col_ch, col_tb = st.columns([1.2, 1])

            with col_ch:
                colors_map = {
                    "SVM": "#1a56db",
                    "Random Forest": "#0ea5e9",
                    "KNN": "#8b5cf6",
                    "Logistic Regression": "#06b6d4",
                    "CNN": "#3b82f6",
                }
                fig = go.Figure()
                for _, row in df.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row["model"]],
                        y=[row["accuracy"]],
                        text=[f"{row['accuracy']}%"],
                        textposition="outside",
                        marker_color=colors_map.get(row["model"], "#1a56db"),
                    ))
                fig.update_layout(
                    paper_bgcolor="#0a1628",
                    plot_bgcolor="#0a1628",
                    font_color="#e2e8f0",
                    showlegend=False,
                    height=300,
                    margin=dict(t=30, b=10, l=10, r=10),
                    yaxis=dict(range=[0, 110], gridcolor="#1e3a5f"),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_tb:
                df_s = df[["model", "accuracy", "f1",
                            "precision", "recall"]].copy()
                df_s.columns = ["Model", "Accuracy",
                                 "F1", "Precision", "Recall"]
                st.dataframe(
                    df_s.style.highlight_max(
                        subset=["Accuracy", "F1", "Precision", "Recall"],
                        color="#1a56db33",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("🔔 Train models first to see results here.")

    with sys_tab2:
        genuine_dir = os.path.join(DATASET_DIR, "genuine")
        forged_dir = os.path.join(DATASET_DIR, "forged")
        dataset_ok = os.path.isdir(genuine_dir) and os.path.isdir(forged_dir)

        if not dataset_ok:
            st.error("✗ Run: python preprocess_dataset.py first")
        else:
            g = len([f for f in os.listdir(genuine_dir) if f.endswith(".png")])
            f = len([f for f in os.listdir(forged_dir) if f.endswith(".png")])
            c1, c2, c3 = st.columns(3)
            c1.success("✔ Dataset Ready")
            c2.metric("Genuine", g)
            c3.metric("Forged", f)

            st.markdown("---")
            for icon, name, desc in [
                ("🧠", "CNN", "Deep Learning – Convolutional Neural Network"),
                ("⚙", "SVM", "Support Vector Machine with RBF kernel"),
                ("⚙", "Random Forest", "Ensemble of 200 decision trees"),
                ("⚙", "KNN", "K-Nearest Neighbours (k=5)"),
                ("⚙", "Logistic Regression",
                 "Linear classifier with L2 regularisation"),
            ]:
                c1, c2 = st.columns([0.3, 5])
                c1.markdown(
                    f"<div style='font-size:1.3rem;'>{icon}</div>",
                    unsafe_allow_html=True,
                )
                c2.markdown(f"**{name}** — {desc}")

            st.markdown("---")

            if st.button(
                "⚙ Start Training All Models",
                use_container_width=True,
                key="train_btn",
            ):
                from sklearn.model_selection import train_test_split
                prog = st.progress(0)
                status = st.empty()
                try:
                    status.info("📂 Extracting features...")
                    prog.progress(10)
                    X, y = build_dataset(DATASET_DIR)
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    res = []

                    for name, fn, pct in [
                        ("SVM", train_svm, 25),
                        ("Random Forest", train_random_forest, 45),
                        ("KNN", train_knn, 60),
                        ("Logistic Regression", train_logistic_regression, 75),
                    ]:
                        status.info(f"🔄 Training {name}...")
                        r = fn(X_tr, y_tr, X_te, y_te)
                        res.append(r)
                        prog.progress(pct)
                        st.success(f"✔ {name} — {r['accuracy']}%")

                    status.info("🧠 Training CNN...")
                    prog.progress(82)
                    try:
                        Xc, yc = build_cnn_dataset(DATASET_DIR)
                        Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
                            Xc, yc, test_size=0.2, random_state=42, stratify=yc
                        )
                        cnn_r = train_cnn(Xc_tr, yc_tr, Xc_te, yc_te)
                    except Exception as cnn_err:
                        cnn_r = {
                            "model": "CNN", "accuracy": 0,
                            "precision": 0, "recall": 0, "f1": 0,
                            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                            "train_time": 0,
                            "report": str(cnn_err),
                            "error": str(cnn_err),
                        }
                    res.append(cnn_r)
                    prog.progress(100)
                    status.success("✅ All models trained!")

                    out = {
                        "results": res,
                        "dataset_info": {
                            "total": int(len(y)),
                            "genuine": int((y == 0).sum()),
                            "forged": int((y == 1).sum()),
                            "features": int(X.shape[1]),
                        },
                    }
                    with open(
                        os.path.join(REPORTS_DIR, "training_results.json"), "w"
                    ) as ff:
                        json.dump(out, ff, indent=2)

                    df = pd.DataFrame(res)[
                        ["model", "accuracy", "precision",
                         "recall", "f1", "train_time"]
                    ]
                    df.columns = [
                        "Model", "Accuracy%", "Precision%",
                        "Recall%", "F1%", "Time(s)"
                    ]
                    st.dataframe(df, use_container_width=True, hide_index=True)

                except Exception as e:
                    status.error(f"❌ {e}")


# ── Logout ────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
_, _, col_out = st.columns([5, 2, 1])
with col_out:
    if st.button("🚪 Logout", key="logout_btn"):
        st.session_state.logged_in = False
        st.rerun()
