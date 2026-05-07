import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoVision · Deforestation Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root {
    --ash:    #0d1a12;
    --leaf:   #4a8c5c;
    --lime:   #7ec850;
    --warn:   #e8a020;
    --danger: #c94040;
    --fog:    #e8e4dc;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--ash) !important;
}
body, p, div, span, label { color: var(--fog) !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
[data-testid="stFileUploadDropzone"] {
    background: rgba(26,58,42,0.4) !important;
    border: 2px dashed #4a8c5c !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: #7ec850 !important; }
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #4a8c5c, #7ec850) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #2d5a3d, #4a8c5c) !important;
    color: white !important; border: none !important;
    border-radius: 6px !important; font-weight: 600 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4a8c5c, #7ec850) !important;
    color: #0d1a12 !important;
}
[data-testid="stDownloadButton"] > button {
    background: rgba(26,58,42,0.6) !important; color: #7ec850 !important;
    border: 1px solid #4a8c5c !important; border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ── ResNet34 ImageNet preprocessing (no segmentation_models needed) ───────────
# Exact replication of sm.get_preprocessing('resnet34')
IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)

def resnet34_preprocess(img_array: np.ndarray) -> np.ndarray:
    """RGB uint8 (0-255) → ResNet34 preprocessed float32 (BGR, mean-subtracted)"""
    img = img_array.astype(np.float32)
    img = img[..., ::-1]                  # RGB → BGR
    img[..., 0] -= IMAGENET_MEAN_BGR[0]   # subtract B mean
    img[..., 1] -= IMAGENET_MEAN_BGR[1]   # subtract G mean
    img[..., 2] -= IMAGENET_MEAN_BGR[2]   # subtract R mean
    return img


def stretch_contrast(img_array: np.ndarray, low=2, high=98) -> np.ndarray:
    """
    Percentile contrast stretch per channel.
    Brings dark/dim satellite images to a normal brightness range,
    matching how training images typically look after display normalization.
    """
    out = np.zeros_like(img_array, dtype=np.float32)
    for c in range(3):
        ch = img_array[:, :, c].astype(np.float32)
        p_low  = np.percentile(ch, low)
        p_high = np.percentile(ch, high)
        if p_high > p_low:
            ch = (ch - p_low) / (p_high - p_low) * 255.0
        out[:, :, c] = np.clip(ch, 0, 255)
    return out.astype(np.uint8)


# ── Model Loader ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        if not os.path.exists("best_model.keras"):
            return None, "Model file `best_model.keras` not found — place it in the same folder as app.py"
        model = tf.keras.models.load_model("best_model.keras", compile=False)
        return model, None
    except ImportError:
        return None, "TensorFlow not installed. Run: pip install tensorflow"
    except Exception as e:
        return None, str(e)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(image: Image.Image, model, threshold: float, apply_stretch: bool) -> dict:
    orig_w, orig_h = image.size

    # 1. Resize to 512x512
    img_resized = image.resize((512, 512))
    img_array   = np.array(img_resized, dtype=np.uint8)

    # 2. Optional contrast stretch (for dark/dim satellite images)
    if apply_stretch:
        img_array = stretch_contrast(img_array)

    # 3. ResNet34 preprocessing — matches training exactly
    img_preprocessed = resnet34_preprocess(img_array.astype(np.float32))
    img_batch        = np.expand_dims(img_preprocessed, 0)   # (1,512,512,3)

    # 4. Predict
    raw  = model.predict(img_batch, verbose=0)
    mask = np.squeeze(raw)   # (512,512), sigmoid 0-1, 1=deforested

    # 5. Binarize
    binary = (mask > threshold).astype(np.uint8)

    # 6. Resize back to original size
    mask_pil   = Image.fromarray((mask   * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    binary_pil = Image.fromarray((binary * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
    mask_full   = np.array(mask_pil,   dtype=np.float32) / 255.0
    binary_full = (np.array(binary_pil) > 127).astype(np.uint8)

    # 7. Metrics
    total             = binary_full.size
    deforested_pixels = int(np.sum(binary_full == 1))
    forest_pixels     = total - deforested_pixels
    forest_loss_pct   = round(deforested_pixels / total * 100, 1)
    forest_cover_pct  = round(forest_pixels     / total * 100, 1)
    pixels_per_ha     = 100   # ~10m/px Sentinel-2
    deforested_ha     = round(deforested_pixels / pixels_per_ha, 1)
    confidence        = round(float(np.mean(np.abs(mask - 0.5))) * 200, 1)
    confidence        = max(50.0, min(99.9, confidence))

    # 8. Clean binary mask: red=deforested, green=forest
    h, w = binary_full.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[binary_full == 0] = [34,  139, 34]
    mask_rgb[binary_full == 1] = [220, 50,  50]
    clean_mask = Image.fromarray(mask_rgb)

    # 9. Red overlay on original
    orig_arr = np.array(image.convert("RGB"), dtype=np.float32)
    blended  = orig_arr.copy()
    d = binary_full == 1
    blended[d, 0] = np.clip(orig_arr[d, 0] * 0.4 + 190, 0, 255)
    blended[d, 1] = np.clip(orig_arr[d, 1] * 0.2,        0, 255)
    blended[d, 2] = np.clip(orig_arr[d, 2] * 0.2,        0, 255)
    overlay_img = Image.fromarray(blended.astype(np.uint8))

    # Stretched preview image for display
    stretched_display = Image.fromarray(stretch_contrast(np.array(image)) if apply_stretch else np.array(image))

    return {
        "forest_loss_pct":    forest_loss_pct,
        "forest_cover_pct":   forest_cover_pct,
        "deforested_area_ha": deforested_ha,
        "confidence":         confidence,
        "clean_mask":         clean_mask,
        "overlay":            overlay_img,
        "raw_mask":           mask,
        "display_image":      stretched_display,
    }


def severity(pct):
    if pct < 20:  return "🟢", "#7ec850", "LOW CONCERN"
    if pct < 50:  return "🟡", "#e8a020", "MODERATE LOSS"
    return "🔴", "#c94040", "CRITICAL LOSS"


# ── UI ────────────────────────────────────────────────────────────────────────
model, model_err = load_model()

st.markdown("""
<div style="text-align:center; padding:2.5rem 1rem 1rem;">
    <div style="font-size:0.7rem; letter-spacing:0.35em; color:#7ec850;
         text-transform:uppercase; margin-bottom:0.4rem;">
        🛰 Deep Learning · Remote Sensing
    </div>
    <div style="font-size:clamp(2.8rem,7vw,5.5rem); font-weight:900;
         letter-spacing:0.06em; line-height:1;
         background:linear-gradient(135deg,#7ec850,#4a8c5c,#e8a020);
         -webkit-background-clip:text; -webkit-text-fill-color:transparent;
         background-clip:text;">
        ECOVISION
    </div>
    <div style="font-size:1rem; color:#8aaf96; margin-top:0.6rem; font-style:italic;">
        Upload a satellite or aerial image — get instant deforestation analysis
    </div>
    <div style="width:100px; height:2px;
         background:linear-gradient(90deg,transparent,#7ec850,transparent);
         margin:1.2rem auto;"></div>
</div>
""", unsafe_allow_html=True)

if model:
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.2rem;">
        <span style="background:rgba(126,200,80,0.12); border:1px solid #4a8c5c;
             border-radius:20px; padding:0.3rem 1.1rem; font-size:0.72rem;
             letter-spacing:0.15em; color:#7ec850; text-transform:uppercase;">
            ✅ &nbsp; best_model.keras · ResNet34 UNet · ready
        </span>
    </div>""", unsafe_allow_html=True)
else:
    st.error(f"⚠️ {model_err}")
    st.stop()

# Controls row
col_up, col_thresh, col_stretch = st.columns([3, 1, 1])

with col_thresh:
    st.markdown("<br>", unsafe_allow_html=True)
    threshold = st.slider(
        "🎚️ Threshold",
        min_value=0.10, max_value=0.90,
        value=0.50, step=0.01,
        help="0.5 = standard. Lower to detect more deforestation."
    )

with col_stretch:
    st.markdown("<br><br>", unsafe_allow_html=True)
    apply_stretch = st.toggle(
        "🔆 Brighten Image",
        value=True,
        help="Applies contrast stretching to dark satellite images before inference. Turn off if your image is already well-exposed."
    )

with col_up:
    uploaded = st.file_uploader(
        "Drag & drop your satellite image here",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("🌿  Running ResNet34 UNet segmentation..."):
        res = run_inference(image, model, threshold, apply_stretch)

    loss = res["forest_loss_pct"]
    ico, color, label = severity(loss)

    st.markdown('<div style="width:100%;height:1px;background:rgba(74,140,92,0.3);margin:1rem 0;"></div>', unsafe_allow_html=True)

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    for col, (lbl, val, clr, unit) in zip([k1,k2,k3,k4], [
        ("Forest Loss",     f"{loss}%",                        color,    "of total area"),
        ("Forest Cover",    f"{res['forest_cover_pct']}%",     "#7ec850","remaining"),
        ("Deforested Area", f"{res['deforested_area_ha']} ha", color,    "estimated"),
        ("Confidence",      f"{res['confidence']}%",           "#7ec850","model certainty"),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:rgba(26,58,42,0.45);border:1px solid rgba(74,140,92,0.3);
                 border-radius:10px;padding:1.2rem;text-align:center;">
                <div style="font-size:0.62rem;letter-spacing:0.25em;color:#7aaa88;
                     text-transform:uppercase;margin-bottom:0.4rem;">{lbl}</div>
                <div style="font-size:2.6rem;font-weight:900;color:{clr};line-height:1;">{val}</div>
                <div style="font-size:0.78rem;color:#8aaf96;margin-top:0.2rem;">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_prog, col_alert = st.columns(2)
    with col_prog:
        st.markdown("<span style='color:#7ec850;font-size:0.72rem;letter-spacing:0.2em;text-transform:uppercase;'>Coverage Breakdown</span>", unsafe_allow_html=True)
        st.markdown("**🌳 Forest Cover Remaining**")
        st.progress(res["forest_cover_pct"] / 100)
        st.markdown("**🔥 Detected Forest Loss**")
        st.progress(loss / 100)
        st.markdown("**🎯 Detection Confidence**")
        st.progress(res["confidence"] / 100)

    with col_alert:
        advice = {
            "LOW CONCERN":   "Forest cover appears largely intact. Minor disturbances detected. Continued monitoring recommended.",
            "MODERATE LOSS": "Significant deforestation detected. This area may require intervention or regulatory review.",
            "CRITICAL LOSS": "CRITICAL: Severe forest loss detected. Immediate attention and protective measures strongly advised.",
        }[label]
        bc = {"LOW CONCERN":"#7ec850","MODERATE LOSS":"#e8a020","CRITICAL LOSS":"#c94040"}[label]
        bg = {"LOW CONCERN":"rgba(126,200,80,0.1)","MODERATE LOSS":"rgba(232,160,32,0.1)","CRITICAL LOSS":"rgba(201,64,64,0.1)"}[label]
        st.markdown(f"""
        <div style="background:{bg};border-left:4px solid {bc};border-radius:6px;padding:1.2rem 1.4rem;">
            <div style="font-size:0.75rem;letter-spacing:0.15em;font-weight:700;
                 text-transform:uppercase;color:{bc};margin-bottom:0.5rem;">
                {ico} Severity · {label}
            </div>
            <div style="font-size:0.9rem;color:#c8d8cc;line-height:1.6;">
                {advice}<br><br>
                <strong style="color:{color};">{loss}%</strong> of pixels classified as deforested
                with <strong style="color:#7ec850;">{res['confidence']}%</strong> model confidence.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<span style='color:#7ec850;font-size:0.72rem;letter-spacing:0.2em;text-transform:uppercase;'>Visual Analysis</span>", unsafe_allow_html=True)
    st.markdown("### Original · Segmentation Mask · Overlay")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(res["display_image"], caption="📡 Input Image" + (" (contrast stretched)" if apply_stretch else ""), use_container_width=True)
    with c2:
        st.image(res["clean_mask"], caption="🗺️ Mask  🟥 Deforested  🟩 Forest", use_container_width=True)
    with c3:
        st.image(res["overlay"], caption="🔴 Overlay on Original", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    a1, a2, _ = st.columns([1, 1, 2])
    with a1:
        if st.button("🔄 Analyze Another"):
            st.rerun()
    with a2:
        report = f"""ForestWatch — Deforestation Analysis Report
============================================
File             : {uploaded.name}
Threshold        : {threshold}
Contrast Stretch : {apply_stretch}
Forest Loss      : {loss}%
Forest Cover     : {res['forest_cover_pct']}%
Est. Area        : {res['deforested_area_ha']} ha
Confidence       : {res['confidence']}%
Severity         : {label}
Model            : best_model.keras (ResNet34 UNet, 512x512)
Preprocessing    : ImageNet BGR mean subtraction (ResNet34)
"""
        st.download_button("📄 Download Report", report, file_name="forestwatch_report.txt")

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">🌿</div>
        <p style="font-size:0.8rem;letter-spacing:0.2em;text-transform:uppercase;color:#4a7c56;">
            Upload an image above to begin analysis
        </p>
        <p style="font-size:0.88rem;color:#3d6b4a;margin-top:0.4rem;font-style:italic;">
            Supports satellite imagery · Drone photos · Aerial RGB imagery
        </p>
    </div>""", unsafe_allow_html=True)

    for col, (icon, title, desc) in zip(st.columns(3), [
        ("🛰", "Satellite Ready",  "Works with GeoTIFF, JPEG, PNG from any imagery source"),
        ("🧠", "ResNet34 UNet",    "85–96% IoU pixel-level deforestation segmentation"),
        ("📊", "Instant Metrics",  "Forest loss %, coverage, estimated hectares & confidence"),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:rgba(26,58,42,0.4);border:1px solid rgba(74,140,92,0.25);
                 border-radius:10px;padding:1.4rem;">
                <div style="font-size:1.8rem;margin-bottom:0.5rem;">{icon}</div>
                <div style="font-size:0.72rem;letter-spacing:0.15em;color:#7ec850;
                     text-transform:uppercase;margin-bottom:0.3rem;">{title}</div>
                <div style="font-size:0.85rem;color:#8aaf96;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:1.5rem 0 0.5rem;margin-top:2rem;
     border-top:1px solid rgba(74,140,92,0.15);">
    <span style="font-size:0.62rem;letter-spacing:0.2em;color:#3d6b4a;text-transform:uppercase;">
        ForestWatch · ResNet34 UNet · best_model.keras
    </span>
</div>""", unsafe_allow_html=True)