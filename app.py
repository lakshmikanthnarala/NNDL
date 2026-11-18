from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from predict import load_predictor, predict_image


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sign_language_model.keras"
OUTPUTS = ROOT / "outputs"

st.set_page_config(
    page_title="SignSpeak | ASL Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    :root {
      --base:#0c1015; --sidebar:#090d12; --panel:#151a20; --panel-raised:#1a2027;
      --line:#252d35; --cyan:#39d9ef; --cyan-soft:#a3f4fc; --text:#e9eef2;
      --muted:#89949e; --green:#77e4c1; --red:#ff8f91;
    }
    .stApp { background:var(--base); color:var(--text); }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { color:#f3f6f7 !important; letter-spacing:-.035em; }
    h1 { font-size:2rem !important; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stMain"] .block-container { max-width:1600px; padding:1.55rem 2rem 3rem; }
    [data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid #1d242b; }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--muted); }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color:#b8c1c8; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { color:var(--cyan); }
    .brand-lockup { display:flex; align-items:center; gap:11px; margin:4px 0 2px; }
    .brand-mark { display:grid; place-items:center; width:34px; height:34px; border:1px solid #277f89; border-radius:7px; color:var(--cyan); background:#102126; font:700 12px 'Space Mono',monospace; }
    .brand-name { color:#f0f4f5; font-weight:700; letter-spacing:.02em; }
    .brand-caption { color:#77838c; font:9px 'Space Mono',monospace; letter-spacing:.1em; margin-top:3px; }
    [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:15px 18px; border-radius:8px; }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    [data-testid="stMetricValue"] { color:var(--cyan); font-family:'Space Mono',monospace; }
    [data-testid="stTabs"] [role="tablist"] { gap:8px; border-bottom:1px solid var(--line); }
    [data-testid="stTabs"] button[role="tab"] { color:#9ba6af; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:var(--cyan); border-bottom-color:var(--cyan); }
    [data-testid="stFileUploader"], [data-testid="stCameraInput"] { background:#10151a; border:1px solid var(--line); border-radius:8px; }
    [data-testid="stImage"] img { border-radius:6px; }
    [data-testid="stAlert"] { border-radius:8px; }
    .hero { position:relative; overflow:hidden; padding:28px 30px; border:1px solid #2a3940; border-radius:10px; background:linear-gradient(112deg,#17232a 0%,#11171d 68%,#14232a 100%); margin-bottom:20px; }
    .hero:after { content:""; position:absolute; width:220px; height:220px; right:6%; top:-135px; border:1px solid #2b6971; border-radius:50%; box-shadow:0 0 0 24px #172a2c,0 0 0 25px #23434a,0 0 0 50px #152327,0 0 0 51px #20353a; opacity:.5; }
    .hero h1 { font-size:2.6rem !important; margin:.4rem 0 !important; }
    .eyebrow { color:var(--cyan); text-transform:uppercase; letter-spacing:.13em; font-family:'Space Mono',monospace; font-weight:700; font-size:.68rem; }
    .muted { color:#a4afb7; }
    .card { padding:19px; border-radius:8px; border:1px solid var(--line); background:var(--panel); min-height:130px; }
    .card h3 { margin:.55rem 0 .35rem; font-size:1.12rem; }
    .status-strip { display:flex; flex-wrap:wrap; gap:0; align-items:center; background:#11161b; border:1px solid var(--line); border-radius:8px; margin:0 0 18px; }
    .status-item { flex:1; min-width:150px; padding:12px 16px; border-right:1px solid var(--line); }
    .status-item:last-child { border-right:0; }
    .status-label { display:block; margin-bottom:5px; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; font:10px 'Space Mono',monospace; }
    .status-value { color:#e4ebef; font:12px 'Space Mono',monospace; }
    .status-value.accent { color:var(--cyan); }
    .section-label { color:#9ba6af; font:11px 'Space Mono',monospace; text-transform:uppercase; letter-spacing:.1em; }
    .result-panel { min-height:282px; padding:20px; border:1px solid var(--line); border-radius:8px; background:var(--panel); }
    .result-letter { color:var(--cyan); font:700 5rem 'Space Mono',monospace; line-height:1; }
    .result-empty { display:flex; min-height:190px; align-items:center; justify-content:center; text-align:center; color:var(--muted); border:1px dashed #303a42; border-radius:7px; padding:20px; }
    div.stButton > button, div.stDownloadButton > button { border-radius:6px; border:1px solid #2599a6; background:#087c88; color:#f2feff; font-weight:700; }
    div.stButton > button:hover, div.stDownloadButton > button:hover { border-color:var(--cyan); background:#0995a3; color:white; }
    div.stButton > button:disabled { border-color:#303940; background:#1b2228; color:#65717a; }
    code { color:var(--cyan-soft) !important; background:#10151a !important; }
    @media (max-width: 760px) {
      [data-testid="stMain"] .block-container { padding:1rem 1rem 2rem; }
      .hero { padding:23px 20px; }
      .hero h1 { font-size:2.15rem !important; }
      .status-item { min-width:50%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained SignSpeak model...")
def get_predictor():
    return load_predictor(MODEL_PATH)


def render_prediction(image: Image.Image, key: str) -> None:
    st.session_state["latest_prediction"] = None
    st.session_state["latest_confidence"] = None
    st.session_state["latest_processed"] = None
    try:
        predictor = get_predictor()
        letter, confidence, processed = predict_image(image, predictor)
    except FileNotFoundError:
        st.error(
            "The trained model is not available yet. Put the Kaggle CSV files in "
            "`dataset/` and run `python train_model.py` first."
        )
        return
    except (ValueError, RuntimeError) as error:
        st.error(f"Could not recognize this image: {error}")
        return

    if confidence < 0.25:
        st.session_state["latest_processed"] = processed
        st.error(
            f"The model is uncertain (highest class score: {confidence:.1%}). "
            "Try a clear, centered handshape against a plain background."
        )
        return
    st.session_state["latest_prediction"] = letter
    st.session_state["latest_confidence"] = confidence
    st.session_state["latest_processed"] = processed
    st.session_state["prediction_source"] = key


def page_home() -> None:
    if MODEL_PATH.exists():
        project_summary = (
            "SignSpeak uses a TensorFlow/Keras convolutional neural network trained "
            "on Sign Language MNIST. Predictions come from its saved weights."
        )
    else:
        project_summary = (
            "SignSpeak is built around a TensorFlow/Keras CNN. Add the Kaggle CSV "
            "files and train it to create the saved model used for predictions."
        )
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">SIGN SPEAK&nbsp; / &nbsp;VISION WORKSPACE</div>
          <h1>Make every handshape understood.</h1>
          <p style="font-size:1.02rem;color:#b5c1c8;max-width:610px">
            A focused ASL alphabet studio. Capture one sign, review the model's confidence,
            then choose what becomes part of your message.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ready_label = "● MODEL READY" if MODEL_PATH.exists() else "● MODEL NOT TRAINED"
    ready_class = "accent" if MODEL_PATH.exists() else ""
    st.markdown(
        f"""
        <div class="status-strip">
          <div class="status-item"><span class="status-label">System</span><span class="status-value {ready_class}">{ready_label}</span></div>
          <div class="status-item"><span class="status-label">Recognition scope</span><span class="status-value">24 STATIC LETTERS</span></div>
          <div class="status-item"><span class="status-label">Input</span><span class="status-value">WEBCAM / IMAGE UPLOAD</span></div>
          <div class="status-item"><span class="status-label">Output</span><span class="status-value">USER-CONTROLLED TEXT</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    for col, number, title, copy in [
        (a, "01 / INPUT", "Capture a sign", "Use a webcam snapshot or upload a clear handshape image."),
        (b, "02 / INFERENCE", "Review the result", "See the predicted letter, confidence, and model input preview."),
        (c, "03 / COMPOSE", "Build your message", "Add recognized letters to your text only when you choose."),
    ]:
        with col:
            st.markdown(
                f'<div class="card"><span class="eyebrow">{number}</span>'
                f"<h3>{title}</h3><p class='muted'>{copy}</p></div>",
                unsafe_allow_html=True,
            )
    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Supported alphabet")
        st.markdown(
            "**A B C D E F G H I** &nbsp; · &nbsp; **K L M N O P Q R S T U V W X Y**"
        )
        st.info("J and Z are not included: their ASL signs require movement.")
    with right:
        st.subheader("Project overview")
        st.markdown(
            f"{project_summary} The target is 24 static alphabet classes; "
            "predictions and evaluation scores are never hardcoded."
        )
    if st.button("Start translation", type="primary"):
        st.session_state["navigate_to_translation"] = True
        st.rerun()


def page_translate() -> None:
    model_ready = MODEL_PATH.exists()
    model_status = "MODEL READY" if model_ready else "MODEL NOT TRAINED"
    status_class = "accent" if model_ready else ""
    st.markdown(
        """
        <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:12px;margin-bottom:16px">
          <div><div class="eyebrow">VISION WORKSPACE / SINGLE-FRAME MODE</div>
          <h1 style="margin:.35rem 0">Live sign recognition</h1>
          <div class="muted">Capture or upload one handshape to run ASL alphabet inference.</div></div>
          <div class="section-label">● &nbsp;SESSION ACTIVE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="status-strip">
          <div class="status-item"><span class="status-label">System status</span><span class="status-value {status_class}">● &nbsp;{model_status}</span></div>
          <div class="status-item"><span class="status-label">Model</span><span class="status-value">CNN / 24 CLASSES</span></div>
          <div class="status-item"><span class="status-label">Input format</span><span class="status-value">28 × 28 GRAYSCALE</span></div>
          <div class="status-item"><span class="status-label">Low-confidence gate</span><span class="status-value">&lt; 25% REJECTED</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_col, result_col = st.columns([1.55, 1], gap="medium")
    with input_col:
        st.markdown('<div class="section-label">01 / IMAGE INPUT</div>', unsafe_allow_html=True)
        camera_tab, upload_tab = st.tabs(["WEBCAM SNAPSHOT", "UPLOAD IMAGE"])
        with camera_tab:
            capture = st.camera_input("Position one hand clearly in the frame")
            if capture is not None and st.button("Run inference", key="camera_predict", use_container_width=True):
                render_prediction(Image.open(capture).convert("RGB"), "camera")
        with upload_tab:
            uploaded = st.file_uploader(
                "Choose a PNG, JPG, or JPEG image",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=False,
            )
            if uploaded is not None:
                try:
                    image = Image.open(uploaded).convert("RGB")
                    st.image(image, caption="Input image", use_container_width=True)
                    if st.button("Run inference", key="upload_predict", use_container_width=True):
                        render_prediction(image, "upload")
                except (OSError, ValueError) as error:
                    st.error(f"This image could not be opened: {error}")

    with result_col:
        st.markdown('<div class="section-label">02 / CLASSIFIER OUTPUT</div><br>', unsafe_allow_html=True)
        letter = st.session_state.get("latest_prediction")
        confidence = st.session_state.get("latest_confidence")
        processed = st.session_state.get("latest_processed")
        if letter:
            st.markdown(
                f"""
                <div class="result-panel">
                  <div class="eyebrow">PREDICTED ASL LETTER</div>
                  <div class="result-letter">{letter}</div>
                  <div style="color:#a7b3bb;margin-top:8px">Static alphabet class</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(confidence, text=f"Model confidence · {confidence:.1%}")
            if processed is not None:
                st.image(processed, caption="Normalized 28 × 28 model input", width=160)
            st.caption(
                "Confidence is the CNN softmax score, not a guarantee of correctness. "
                "This model recognizes isolated static signs only."
            )
        elif processed is not None:
            st.markdown(
                '<div class="result-empty">Low confidence — try a centered handshape '
                'against a plain, well-lit background.</div>',
                unsafe_allow_html=True,
            )
            st.image(processed, caption="Normalized 28 × 28 model input", width=160)
        else:
            st.markdown(
                '<div class="result-empty">Your prediction will appear here after you '
                'capture or upload a handshape.</div>',
                unsafe_allow_html=True,
            )
        if letter:
            st.markdown('<div class="section-label" style="margin-top:22px">03 / NEXT STEP</div>', unsafe_allow_html=True)
            st.caption("When you are ready, add this letter in Text Builder. Nothing is appended automatically.")


def page_text_builder() -> None:
    st.title("Text builder")
    st.write("Choose when a recognized letter is added; predictions are never appended automatically.")
    letter = st.session_state.get("latest_prediction")
    confidence = st.session_state.get("latest_confidence")
    if letter:
        st.info(f"Latest CNN prediction: **{letter}**" + (f" ({confidence:.1%})" if confidence is not None else ""))
    else:
        st.info("Make a prediction on the Live & Upload page to get started.")

    st.session_state.setdefault("built_text", "")
    add_col, space_col, delete_col, clear_col = st.columns(4)
    with add_col:
        if st.button("Add letter", disabled=letter is None, use_container_width=True):
            st.session_state["built_text"] += letter
    with space_col:
        if st.button("Add space", use_container_width=True):
            st.session_state["built_text"] += " "
    with delete_col:
        if st.button("Delete last character", use_container_width=True):
            st.session_state["built_text"] = st.session_state["built_text"][:-1]
    with clear_col:
        if st.button("Clear text", use_container_width=True):
            st.session_state["built_text"] = ""

    st.subheader("Your text")
    text = st.session_state["built_text"]
    st.code(text or "Your text will appear here", language=None)
    if text:
        st.download_button(
            "Download text",
            data=text,
            file_name="signspeak_text.txt",
            mime="text/plain",
        )
        safe_text = json.dumps(text)
        components.html(
            f"""
            <button onclick='copyText()' style="background:#1767c7;color:white;border:1px solid #398bff;
              border-radius:9px;padding:10px 16px;font-weight:700;cursor:pointer">Copy generated text</button>
            <span id="copy-status" style="color:#aab8ca;margin-left:10px"></span>
            <script>
              const textToCopy = {safe_text};
              async function copyText() {{
                try {{
                  await navigator.clipboard.writeText(textToCopy);
                  document.getElementById("copy-status").textContent = "Copied";
                }} catch (e) {{
                  const field = document.createElement("textarea");
                  field.value = textToCopy; document.body.appendChild(field);
                  field.select(); document.execCommand("copy"); field.remove();
                  document.getElementById("copy-status").textContent = "Copied";
                }}
              }}
            </script>
            """,
            height=48,
        )


def page_performance() -> None:
    st.title("Model performance")
    metrics_path = OUTPUTS / "evaluation_results.json"
    if not metrics_path.exists():
        st.warning(
            "No evaluation results exist yet. Train the model with the supplied "
            "dataset to generate measured metrics and charts."
        )
        return
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    train_col, test_col, class_col = st.columns(3)
    train_col.metric("Training accuracy", f"{metrics['training_accuracy']:.2%}")
    test_col.metric("Testing accuracy", f"{metrics['testing_accuracy']:.2%}")
    class_col.metric("Number of classes", str(metrics["number_of_classes"]))
    st.caption(
        f"Validation accuracy: {metrics['validation_accuracy']:.2%} · "
        f"Test loss: {metrics['testing_loss']:.4f} · "
        f"Test examples: {metrics['test_examples']:,}"
    )

    for image_name, caption in [
        ("accuracy.png", "Training and validation accuracy"),
        ("loss.png", "Training and validation loss"),
        ("confusion_matrix.png", "Test-set confusion matrix"),
    ]:
        image_path = OUTPUTS / image_name
        if image_path.exists():
            st.image(str(image_path), caption=caption, use_container_width=True)

    report_path = OUTPUTS / "classification_report.txt"
    if report_path.exists():
        with st.expander("Classification report"):
            st.code(report_path.read_text(encoding="utf-8"), language="text")
    summary_path = OUTPUTS / "model_summary.txt"
    if summary_path.exists():
        with st.expander("CNN architecture"):
            st.code(summary_path.read_text(encoding="utf-8"), language="text")


def page_about() -> None:
    st.title("About SignSpeak")
    sections = [
        ("Problem statement", "People who use ASL can face communication barriers with people unfamiliar with the language. This academic prototype explores static handshape classification."),
        ("Objectives", "Train a genuine CNN on Sign Language MNIST, evaluate it on the provided held-out test set, and make its single-letter predictions accessible through a simple web interface."),
        ("Methodology", "CSV pixel values are reshaped to 28 × 28 grayscale images, normalized, label-remapped, and split into training and validation sets. Mild geometric augmentation is applied during training only."),
        ("CNN architecture", "Two convolution/max-pooling blocks (32 and 64 filters), a 128-unit ReLU dense layer, dropout, and a 24-class softmax output."),
        ("Dataset", "Sign Language MNIST contains 24 static ASL alphabet classes, excluding J and Z. Its integer labels skip 9 for J; the project explicitly maps labels 10–24 to K–Y."),
        ("Limitations", "This model classifies isolated, static images from a small 28 × 28 dataset. It is not a continuous sign-language interpreter, does not detect motion-based signs, and may perform poorly on natural webcam scenes unlike the dataset."),
        ("Future enhancements", "Add hand localization and a broader in-the-wild dataset, temporal models for motion-based signs, user testing, and support for additional sign languages."),
    ]
    for title, body in sections:
        with st.expander(title, expanded=title == "Problem statement"):
            st.write(body)


st.sidebar.markdown(
    '<div class="brand-lockup"><div class="brand-mark">SS</div><div>'
    '<div class="brand-name">SignSpeak</div>'
    '<div class="brand-caption">ASL VISION STUDIO</div></div></div>',
    unsafe_allow_html=True,
)
st.sidebar.write("")
pages = ["Home", "Live & Upload", "Text Builder", "Model Performance", "About Project"]
if st.session_state.pop("navigate_to_translation", False):
    st.session_state["navigation_page"] = "Live & Upload"
if st.session_state.get("navigation_page") not in pages:
    st.session_state["navigation_page"] = "Home"
selected_page = st.sidebar.radio("Navigate", pages, key="navigation_page")
st.sidebar.divider()
st.sidebar.markdown('<div class="section-label">MODEL STATUS</div>', unsafe_allow_html=True)
st.sidebar.caption("● Trained model found" if MODEL_PATH.exists() else "○ Train model to enable predictions")

{
    "Home": page_home,
    "Live & Upload": page_translate,
    "Text Builder": page_text_builder,
    "Model Performance": page_performance,
    "About Project": page_about,
}[selected_page]()
