import os
import uuid
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template, jsonify, session
from PIL import Image
import torchvision.transforms as transforms
from classification_model import DeepCNN, DeepANN, ResNet, VGG, AlexNet
import json
from groq import Groq
from dotenv import load_dotenv
import cv2
import numpy as np
import textwrap
from huggingface_hub import hf_hub_download
import logging

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Environment ────────────────────────────────────────────────
load_dotenv()

# ── Config ─────────────────────────────────────────────────────
HF_REPO_ID   = "Srikanth22MH1A42C6/cataract-classification"   # ← your HuggingFace repo
MODEL_DIR    = "models"                                         # local cache folder
UPLOAD_FOLDER = "static/uploads"

# ── Classes ────────────────────────────────────────────────────
CLASSES = ["Cataract", "Normal"]

MODEL_REGISTRY = {
    "DeepCNN":  DeepCNN,
    "DeepANN":  DeepANN,
    "ResNet":   ResNet,
    "VGG":      VGG,
    "AlexNet":  AlexNet,
}

# Map model name → exact filename on HuggingFace
HF_FILENAMES = {
    "AlexNet": "catarct_or_normalAlexNet.pth",
    "DeepANN": "catarct_or_normalDeepANN.pth",
    "DeepCNN": "catarct_or_normalDeepCNN.pth",
    "ResNet":  "catarct_or_normalResNet.pth",
    "VGG":     "catarct_or_normalVGG.pth",
}

# ── Image transform (identical to training pipeline) ───────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ── Flask app ──────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

os.makedirs(MODEL_DIR,    exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

loaded_models: dict = {}   # in-memory model cache

# ── Thresholds ─────────────────────────────────────────────────
MIN_CONFIDENCE   = 30
MAX_ENTROPY      = 0.67
MAX_LAP_VARIANCE = 8000
ILLUS_HI_SAT_THRESH = 0.75
ILLUS_SKIN_THRESH   = 0.08
WHITE_BG_THRESHOLD  = 0.35
HOUGH_CIRCLE_MAX_MEAN = 185
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tiff"}

# ── Haar cascade ───────────────────────────────────────────────
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# ══════════════════════════════════════════════════════════════
#  MODEL LOADER  — tries local cache first, then HuggingFace
# ══════════════════════════════════════════════════════════════
def get_model(model_name: str):
    """
    Load a model by name.
    Priority:
      1. In-memory cache (fastest)
      2. Local models/ folder
      3. HuggingFace Hub  (auto-downloads + caches locally)
    """
    # 1. Already loaded
    if model_name in loaded_models:
        return loaded_models[model_name]

    if model_name not in MODEL_REGISTRY:
        log.warning("Unknown model requested: %s", model_name)
        return None

    # 2. Determine local path
    hf_filename  = HF_FILENAMES.get(model_name)
    local_path   = os.path.join(MODEL_DIR, hf_filename) if hf_filename else None

    # 3. Download from HuggingFace if not already on disk
    if local_path and not os.path.exists(local_path):
        try:
            log.info("Downloading %s from HuggingFace …", hf_filename)
            downloaded = hf_hub_download(
                repo_id   = HF_REPO_ID,
                filename  = hf_filename,
                local_dir = MODEL_DIR,
            )
            log.info("Downloaded to %s", downloaded)
        except Exception as exc:
            log.error("HuggingFace download failed for %s: %s", model_name, exc)
            return None

    if not local_path or not os.path.exists(local_path):
        log.error("Model file not found: %s", local_path)
        return None

    # 4. Load checkpoint
    try:
        checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            num_classes = checkpoint.get("num_classes", 2)
            model = MODEL_REGISTRY[model_name](num_classes=num_classes)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            # Legacy: raw state dict
            model = MODEL_REGISTRY[model_name](num_classes=2)
            model.load_state_dict(checkpoint, strict=False)

        model.eval()
        loaded_models[model_name] = model
        log.info("Loaded model: %s", model_name)
        return model

    except Exception as exc:
        log.error("Failed to load model %s: %s", model_name, exc)
        return None


def get_available_models() -> list[str]:
    """
    Returns all known model names.
    get_model() will auto-download from HuggingFace if not locally cached.
    """
    return list(HF_FILENAMES.keys())


# ══════════════════════════════════════════════════════════════
#  PREDICTION HELPER  — single image, single model
# ══════════════════════════════════════════════════════════════
def run_inference(model, input_tensor: torch.Tensor) -> dict:
    """
    Run a single forward pass and return structured results.

    input_tensor shape: [1, 3, 128, 128]
      - The leading 1 is the BATCH DIMENSION.
      - PyTorch models always expect (batch, channels, height, width).
      - A lone image is [3, 128, 128]; .unsqueeze(0) adds the batch dim → [1, 3, 128, 128].
    """
    with torch.no_grad():
        output  = model(input_tensor)                              # [1, num_classes]
        probs   = F.softmax(output, dim=1)                        # convert logits → probabilities
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        idx     = torch.argmax(probs, dim=1).item()
        conf    = probs[0][idx].item() * 100

    return {
        "prediction": CLASSES[idx],
        "confidence": round(conf, 2),
        "entropy":    round(entropy, 4),
        "probabilities": {
            "Cataract": round(probs[0][0].item() * 100, 2),
            "Normal":   round(probs[0][1].item() * 100, 2),
        }
    }


# ══════════════════════════════════════════════════════════════
#  REST API ENDPOINT  — /api/predict/<model_name>
# ══════════════════════════════════════════════════════════════
@app.route("/api/predict/<model_name>", methods=["POST"])
def api_predict(model_name: str):
    """
    REST endpoint: POST an eye image, get a prediction back as JSON.

    Usage:
        curl -X POST http://localhost:5000/api/predict/ResNet \
             -F "file=@eye.jpg"

    Returns:
        {
          "model":       "ResNet",
          "prediction":  "Cataract" | "Normal",
          "confidence":  87.43,
          "entropy":     0.1821,
          "probabilities": {"Cataract": 87.43, "Normal": 12.57}
        }
    """
    # ── Validate model name ────────────────────────────────────
    if model_name not in MODEL_REGISTRY:
        return jsonify({
            "error": f"Unknown model '{model_name}'. "
                     f"Available: {list(MODEL_REGISTRY.keys())}"
        }), 400

    # ── Validate file upload ───────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send image as 'file' field."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 415

    # ── Load model ─────────────────────────────────────────────
    model = get_model(model_name)
    if model is None:
        return jsonify({"error": f"Model '{model_name}' could not be loaded."}), 503

    # ── Preprocess ─────────────────────────────────────────────
    try:
        image = Image.open(file.stream).convert("RGB")
        # unsqueeze(0) adds batch dimension: [3,128,128] → [1,3,128,128]
        input_tensor = transform(image).unsqueeze(0)
    except Exception as exc:
        return jsonify({"error": f"Failed to process image: {str(exc)}"}), 422

    # ── Inference ──────────────────────────────────────────────
    result = run_inference(model, input_tensor)
    result["model"] = model_name

    log.info("API /predict/%s → %s (%.1f%%)", model_name, result["prediction"], result["confidence"])
    return jsonify(result), 200


@app.route("/api/predict/ensemble", methods=["POST"])
def api_predict_ensemble():
    """
    Ensemble endpoint: runs ALL available models and returns majority vote.

    Usage:
        curl -X POST http://localhost:5000/api/predict/ensemble \
             -F "file=@eye.jpg"
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    try:
        image = Image.open(file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
    except Exception as exc:
        return jsonify({"error": f"Failed to process image: {str(exc)}"}), 422

    available = get_available_models()
    if not available:
        return jsonify({"error": "No models available."}), 503

    model_results  = []
    cataract_votes = []
    normal_votes   = []

    for name in available:
        model = get_model(name)
        if not model:
            continue
        result = run_inference(model, input_tensor)
        result["model"] = name
        model_results.append(result)
        if result["prediction"] == "Cataract":
            cataract_votes.append(result["confidence"])
        else:
            normal_votes.append(result["confidence"])

    if not model_results:
        return jsonify({"error": "No models could run inference."}), 503

    cataract_count = len(cataract_votes)
    normal_count   = len(normal_votes)

    # Tie → default to Cataract (conservative for medical use)
    final_pred    = "Cataract" if cataract_count >= normal_count else "Normal"
    winning_votes = cataract_votes if final_pred == "Cataract" else normal_votes
    avg_conf      = round(sum(winning_votes) / len(winning_votes), 2) if winning_votes else 0.0
    avg_entropy   = round(sum(r["entropy"] for r in model_results) / len(model_results), 4)

    return jsonify({
        "final_prediction":  final_pred,
        "confidence":        avg_conf,
        "avg_entropy":       avg_entropy,
        "cataract_votes":    cataract_count,
        "normal_votes":      normal_count,
        "models_used":       len(model_results),
        "individual_results": model_results,
    }), 200


@app.route("/api/models", methods=["GET"])
def api_list_models():
    """Returns list of all available (downloaded) models."""
    return jsonify({
        "available_models": get_available_models(),
        "all_models":       list(MODEL_REGISTRY.keys()),
        "repo":             HF_REPO_ID,
    }), 200


# ══════════════════════════════════════════════════════════════
#  GROQ LLM SUMMARY
# ══════════════════════════════════════════════════════════════
def get_groq_summary(final_result, model_results, api_key=None):
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "Groq API Key not provided. Please provide it in the request."
    try:
        client    = Groq(api_key=api_key)
        base_data = (
            f"- Diagnosis: {final_result['prediction']}\n"
            f"- Confidence: {final_result['confidence']:.2f}%\n"
            f"- Support: {final_result['cataract_votes']} out of {final_result['model_count']} models."
        )

        if final_result["prediction"] == "Cataract":
            content_sections = """
## What is Cataract
- A cataract is a clouding of the lens inside your eye.
- It makes your vision look blurry, foggy, or dusty.

## Stages of Cataract
- **Early Stage:** Lens just starting to cloud — vision mostly okay.
- **Mature Stage:** Fully cloudy like thick fog — surgery usually needed.
- **Hypermature Stage:** Long-standing; may cause pain or pressure.

## Causes
- **Aging:** Very common as we get older.
- **Sunlight:** Too much sun without sunglasses.
- **Health Issues:** Diabetes or high blood sugar.
- **Injury:** Past hit or injury to the eye.

## Symptoms
- Blurry or "cloudy" vision.
- Halos around lights at night.
- Colors looking faded or yellow.
- Double vision in one eye.

## Medicine & Free Help
- **Eye Drops:** Keep eyes moist but don't remove the cataract.
- **Free Schemes:** Ayushman Bharat offers Free Cataract Surgery.
- **NGOs:** Lions Club often holds free eye camps.

## Food to Eat
- Green leafy vegetables: Spinach, Methi.
- Orange/Yellow fruits: Carrots, Papaya, Oranges.
- Nuts: Almonds or walnuts daily.

## Surgery Costs (India)
- **Basic (SICS):** Rs.15,000-25,000
- **Advanced (Phaco):** Rs.40,000-80,000
- **Laser/Robot:** Rs.1,00,000+
"""
        else:
            content_sections = """
## Result: Normal & Healthy
- Your scan result is **Normal** — no cataract found.

## Keeping Eyes Healthy
- **Healthy Diet:** Carrots, papayas, leafy greens.
- **Drink Water:** Hydration prevents dry eyes.

## Daily Tips
- **Screen Breaks:** 20-20-20 rule every 20 min.
- **Protection:** Sunglasses on bright days.
- **Sleep:** 7-8 hours protects eye health.

## Stay Proactive
- **Yearly Scan:** Good habit even with normal results.
"""

        pred_label = (
            "Normal (Healthy)" if final_result["prediction"] == "Normal"
            else "Cataract Detected"
        )
        prompt = textwrap.dedent(f"""
            You are a friendly and caring Eye Doctor speaking in plain, simple English.
            Analyze these findings:
            {base_data}

            CRITICAL INSTRUCTIONS:
            1. Start DIRECTLY with: "- **Eye Health Status:**"
            2. Speak in everyday English. Avoid medical jargon.
            3. Use DOUBLE NEWLINES between every header and list item.
            4. Use ONLY Markdown headers (##) and bold (**).
            5. Use BULLET POINTS (-) for everything.

            - **Eye Health Status:** {pred_label}
            - **Neural Support:** {final_result['cataract_votes']}/{final_result['model_count']} models agreed.
            - **Clinical Confidence:** {final_result['confidence']:.2f}%

            {content_sections}
        """).strip()

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=2000,
        )
        return completion.choices[0].message.content
    except Exception as exc:
        return f"Error generating summary: {str(exc)}"


# ══════════════════════════════════════════════════════════════
#  3-TIER EXPLANATION
# ══════════════════════════════════════════════════════════════
def get_cataract_explanation(prediction):
    if prediction != "Cataract":
        return {"simple": "", "technical": "", "ai_model": ""}

    simple = (
        "Yes, this image shows typical cataract signs.\n\n"
        "How we identify cataract from the image:\n\n"
        "1. White / cloudy pupil\n"
        "   Normally the pupil looks black because light enters the eye freely.\n"
        "   In cataract images the pupil area appears milky white.\n\n"
        "2. Loss of transparency\n"
        "   A healthy eye lens is perfectly clear.\n"
        "   Here the centre looks foggy — a major cataract indicator.\n\n"
        "3. Diffuse light reflection\n"
        "   Light reflection spreads across the cloudy lens instead of appearing sharp."
    )
    technical = (
        "In ophthalmology images, cataract is identified by lens opacity patterns.\n\n"
        "Key image features visible:\n\n"
        "- Lens Opacification — central region appears white due to protein aggregation.\n"
        "- Reduced contrast — iris-pupil boundary becomes less distinct.\n"
        "- Scattered illumination — light reflection spreads due to lost transparency.\n"
        "- Central opacity — characteristic of nuclear or mature cataract stages."
    )
    ai_model = (
        "CNN detects cataract using texture and intensity patterns.\n\n"
        "Features extracted:\n"
        "- High pixel-intensity cluster in the pupil region\n"
        "- Reduced dark area (black pupil disappears)\n"
        "- Low edge contrast between iris and lens boundary\n"
        "- Texture irregularity in central lens region\n\n"
        "Pipeline: Image -> Preprocessing -> CNN layers -> FC layer -> Softmax -> Cataract/Normal\n\n"
        "Ensemble (DeepCNN / VGG / ResNet / AlexNet / DeepANN) vote independently; majority decides."
    )
    return {"simple": simple, "technical": technical, "ai_model": ai_model}


# ══════════════════════════════════════════════════════════════
#  HELPER UTILITIES (unchanged)
# ══════════════════════════════════════════════════════════════
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _crop_solid_borders(img, gray, std_thresh=18):
    h, w = gray.shape
    t, b, l, r = 0, h, 0, w
    max_frac = 0.35
    for c in range(int(w * max_frac)):
        if np.std(gray[:, c]) < std_thresh: l = c + 1
        else: break
    for c in range(w - 1, int(w * (1 - max_frac)), -1):
        if np.std(gray[:, c]) < std_thresh: r = c
        else: break
    for row in range(int(h * max_frac)):
        if np.std(gray[row, :]) < std_thresh: t = row + 1
        else: break
    for row in range(h - 1, int(h * (1 - max_frac)), -1):
        if np.std(gray[row, :]) < std_thresh: b = row
        else: break
    if b - t >= 50 and r - l >= 50:
        return img[t:b, l:r], gray[t:b, l:r]
    return img, gray


def _group_dets(dets, prox):
    groups = []
    for d in dets:
        placed = False
        for g in groups:
            gc_x = sum(x[0] for x in g) / len(g)
            gc_y = sum(x[1] for x in g) / len(g)
            if np.hypot(d[0] - gc_x, d[1] - gc_y) < prox:
                g.append(d); placed = True; break
        if not placed:
            groups.append([d])
    return groups


def _group_score(g):  return len(g) * max(d[2] for d in g)
def _group_center(g): return (sum(d[0] for d in g)/len(g), sum(d[1] for d in g)/len(g))


def _near_border(cx, cy, w, h, frac=0.17):
    return (cx < w * frac or cx > w * (1 - frac) or
            cy < h * frac or cy > h * (1 - frac))


def _filter_small_dets(dets, min_size_ratio=0.40):
    if not dets: return dets
    max_s     = max(d[2] for d in dets)
    threshold = max_s * min_size_ratio
    filtered  = [d for d in dets if d[2] >= threshold]
    removed   = len(dets) - len(filtered)
    if removed:
        log.debug("FilterSmall: dropped %d hit(s)", removed)
    return filtered


def _circle_interior_mean(gray, cx, cy, radius):
    h, w = gray.shape
    Y, X  = np.ogrid[:h, :w]
    mask  = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    pixels = gray[mask]
    return float(np.mean(pixels)) if pixels.size > 0 else 255.0


# ══════════════════════════════════════════════════════════════
#  VISUAL FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════
def analyze_eye_features(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path)
        if img is None: return {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx, cy = w // 2, h // 2

        r_pupil = max(10, int(min(h, w) * 0.18))
        r_iris  = max(20, int(min(h, w) * 0.38))

        Y, X = np.ogrid[:h, :w]
        pupil_mask = ((X - cx)**2 + (Y - cy)**2) <= r_pupil**2
        iris_mask  = (((X - cx)**2 + (Y - cy)**2) <= r_iris**2) & ~pupil_mask

        pupil_px = gray[pupil_mask]
        iris_px  = gray[iris_mask]

        pupil_brightness = float(np.mean(pupil_px)) / 255.0 * 100 if pupil_px.size > 0 else 0.0

        if pupil_px.size > 0:
            b_p = img[:,:,0][pupil_mask].astype(float)
            g_p = img[:,:,1][pupil_mask].astype(float)
            r_p = img[:,:,2][pupil_mask].astype(float)
            opacity_score = float(np.mean((r_p > 155) & (g_p > 145) & (b_p > 135))) * 100
        else:
            opacity_score = 0.0

        if pupil_px.size > 0 and iris_px.size > 0:
            diff = abs(float(np.mean(iris_px)) - float(np.mean(pupil_px)))
            iris_contrast = min(100.0, diff / 255.0 * 200.0)
        else:
            iris_contrast = 50.0

        if pupil_px.size > 0:
            std_val   = float(np.std(pupil_px))
            uniformity = max(0.0, 1.0 - std_val / 80.0)
            light_scatter = min(100.0, uniformity * pupil_brightness)
        else:
            light_scatter = 0.0

        return {
            "pupil_brightness": round(pupil_brightness, 1),
            "opacity_score":    round(opacity_score,    1),
            "iris_contrast":    round(iris_contrast,    1),
            "light_scatter":    round(light_scatter,    1),
        }
    except Exception as exc:
        log.error("analyze_eye_features error: %s", exc)
        return {}


# ══════════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════════
def generate_feature_heatmap(image_path: str, output_path: str):
    try:
        img  = cv2.imread(image_path)
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        sigma  = min(h, w) * 0.30
        Y, X   = np.ogrid[:h, :w]
        gauss  = np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*sigma**2)).astype(np.float32)
        feat   = gray.astype(np.float32) / 255.0 * gauss
        feat   = cv2.GaussianBlur(feat, (21, 21), 0)
        if feat.max() > 0: feat = feat / feat.max()
        hm      = cv2.applyColorMap((feat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(img, 0.55, hm, 0.45, 0)
        cv2.imwrite(output_path, blended)
        return output_path
    except Exception as exc:
        log.error("feature_heatmap error: %s", exc)
        return None


def generate_gradcam(model, input_tensor, target_class_idx: int,
                     img_path: str, output_path: str):
    model.eval()
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    if last_conv is None:
        return generate_feature_heatmap(img_path, output_path)

    captured = {}

    def fwd_hook(m, inp, out):
        captured["act"] = out
        out.retain_grad()

    handle = last_conv.register_forward_hook(fwd_hook)
    try:
        inp    = input_tensor.detach().clone()
        output = model(inp)
        handle.remove()
        if "act" not in captured:
            return generate_feature_heatmap(img_path, output_path)
        model.zero_grad()
        output[0, target_class_idx].backward()
        act  = captured["act"].detach().cpu().numpy()[0]
        grad = captured["act"].grad
        if grad is None:
            return generate_feature_heatmap(img_path, output_path)
        grads   = grad.detach().cpu().numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam     = np.einsum("c,chw->hw", weights, act)
        cam     = np.maximum(cam, 0)
        if cam.max() == 0:
            return generate_feature_heatmap(img_path, output_path)
        cam    /= cam.max()
        img_cv  = cv2.imread(img_path)
        if img_cv is None: return None
        h_img, w_img = img_cv.shape[:2]
        cam_up  = cv2.resize(cam, (w_img, h_img))
        hm_img  = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(img_cv, 0.55, hm_img, 0.45, 0)
        cv2.imwrite(output_path, blended)
        return output_path
    except Exception as exc:
        log.error("GradCAM error: %s", exc)
        try: handle.remove()
        except Exception: pass
        return generate_feature_heatmap(img_path, output_path)


# ══════════════════════════════════════════════════════════════
#  IMAGE VALIDATOR (7-layer)
# ══════════════════════════════════════════════════════════════
def is_eye_image(image_path: str) -> tuple:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Unable to read this file. Please upload a valid JPG, PNG, or similar image."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h0, w0 = gray.shape
        log.info("is_eye_image: %s  %dx%d", image_path, w0, h0)

        if h0 < 50 or w0 < 50:
            return False, "Image is too small. Please upload at least 50x50 pixels."

        if np.std(gray) < 5:
            return False, "The image appears blank or solid colour. Please upload a real eye photograph."

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var > MAX_LAP_VARIANCE:
            return False, "This appears to be a screenshot or computer-generated image. Please upload a real photograph."

        hsv_img  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_chan    = hsv_img[:,:,1].ravel().astype(np.float32)
        b_ch = img[:,:,0].ravel().astype(np.int32)
        g_ch = img[:,:,1].ravel().astype(np.int32)
        r_ch = img[:,:,2].ravel().astype(np.int32)

        hi_sat_frac = float(np.mean(s_chan > 200))
        skin_frac   = float(np.mean(
            (r_ch > 80) & (g_ch > 40) & (b_ch > 20) &
            (r_ch - g_ch > 5) & (r_ch - b_ch > 15) &
            (np.maximum(np.maximum(r_ch,g_ch),b_ch) -
             np.minimum(np.minimum(r_ch,g_ch),b_ch) > 15)
        ))
        if (hi_sat_frac > ILLUS_HI_SAT_THRESH) and (skin_frac < ILLUS_SKIN_THRESH):
            return False, "This looks like a cartoon or digital illustration. Please upload a real eye photo."

        white_frac = float(np.mean((r_ch > 235) & (g_ch > 235) & (b_ch > 235)))
        if white_frac > WHITE_BG_THRESHOLD:
            return False, "This does not appear to be a close-up eye image. Please upload a real eye photograph."

        img, gray = _crop_solid_borders(img, gray)
        h, w = gray.shape
        if h < 50 or w < 50:
            return False, "The image appears to be entirely border. Please upload a photo with visible eye content."

        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        min_dim = min(h, w)
        max_dim = max(h, w)

        min_det  = max(25, int(min_dim * 0.07))
        all_dets = []
        for gimg in [gray, gray_eq]:
            for (x, y, ew, eh) in eye_cascade.detectMultiScale(
                    gimg, scaleFactor=1.05, minNeighbors=6, minSize=(min_det, min_det)):
                all_dets.append((x + ew//2, y + eh//2, ew))

        if all_dets:
            all_dets = _filter_small_dets(all_dets, min_size_ratio=0.40)
            prox     = max_dim * 0.40
            groups   = _group_dets(all_dets, prox)
            strong   = [g for g in groups if max(d[2] for d in g) >= min_dim * 0.05]

            if len(strong) == 1:
                return True, ""

            if len(strong) > 1:
                scored = sorted(strong, key=_group_score, reverse=True)
                dom    = scored[0]
                dom_s  = _group_score(dom)
                dom_c  = _group_center(dom)
                interior_sec = [
                    g for g in scored[1:]
                    if not _near_border(*_group_center(g), w, h)
                    and _group_score(g) >= dom_s * 0.40
                ]
                if not interior_sec:
                    return True, ""
                sec   = max(interior_sec, key=_group_score)
                sec_s = _group_score(sec)
                sec_c = _group_center(sec)
                if dom_s >= sec_s * 3.0:
                    return True, ""
                sep = np.hypot(dom_c[0]-sec_c[0], dom_c[1]-sec_c[1])
                if sep > max_dim * 0.65:
                    return False, "This photo appears to show more than one eye. Please upload a close-up of just ONE eye."
                return True, ""

        # Hough fallback
        blurred = cv2.GaussianBlur(gray_eq, (9, 9), 2)
        raw_circles = []
        for param2 in [40, 30, 22, 18, 14]:
            cc = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=int(min_dim*0.30), param1=50, param2=param2,
                minRadius=int(min_dim*0.10), maxRadius=int(min_dim*0.68))
            if cc is not None:
                raw_circles = np.round(cc[0]).astype(int).tolist()
                if len(raw_circles) <= 12: break

        interior = [c for c in raw_circles
                    if w*0.08 <= c[0] <= w*0.92 and h*0.08 <= c[1] <= h*0.92
                    and c[2] >= min_dim*0.10]
        if not interior:
            return False, "No eye could be detected. Please upload a clear, well-lit, front-facing close-up of a single open eye."

        dark_circles = [c for c in interior
                        if _circle_interior_mean(gray, int(c[0]), int(c[1]), int(c[2])) <= HOUGH_CIRCLE_MAX_MEAN]
        if not dark_circles:
            return False, "This does not appear to contain an eye. Please upload a real close-up photograph."

        groups = _group_dets(dark_circles, max_dim * 0.40)
        groups_sorted = sorted(groups, key=lambda g: max(c[2] for c in g), reverse=True)

        if len(groups_sorted) == 1:
            return True, ""

        primary_r   = max(c[2] for c in groups_sorted[0])
        secondary_r = max(c[2] for c in groups_sorted[1])
        if secondary_r < primary_r * 0.70:
            return True, ""

        pc  = _group_center(groups_sorted[0])
        sc  = _group_center(groups_sorted[1])
        sep = np.hypot(pc[0]-sc[0], pc[1]-sc[1])
        if sep > max_dim * 0.65:
            return False, "This photo appears to show more than one eye. Please upload a close-up of just ONE eye."

        return True, ""

    except Exception as exc:
        return False, f"An error occurred while processing the image: {str(exc)}"


# ══════════════════════════════════════════════════════════════
#  MAIN WEB ROUTE
# ══════════════════════════════════════════════════════════════
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_data = None
    error_message   = None
    image_url       = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            error_message = "No file was selected. Please choose an eye image to upload."
            return render_template("index.html", error_message=error_message)

        if not allowed_file(file.filename):
            error_message = "Unsupported file type. Please upload a JPG, PNG, WEBP, or BMP image."
            return render_template("index.html", error_message=error_message)

        ext       = file.filename.rsplit(".", 1)[1].lower()
        safe_name = f"{uuid.uuid4().hex}.{ext}"
        img_path  = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(img_path)
        image_url = "/" + img_path

        groq_api_key = request.form.get("groq_api_key")

        is_valid, err_msg = is_eye_image(img_path)
        if not is_valid:
            try: os.remove(img_path)
            except Exception: pass
            return render_template("index.html", error_message=err_msg, image_url=None)

        image        = Image.open(img_path).convert("RGB")
        # unsqueeze(0): adds batch dimension [3,128,128] → [1,3,128,128]
        input_tensor = transform(image).unsqueeze(0)

        available_models = get_available_models()
        model_results    = []
        cataract_votes   = []
        normal_votes     = []

        for m_name in available_models:
            model = get_model(m_name)
            if not model: continue
            result = run_inference(model, input_tensor)
            result["model"] = m_name
            log.info("Model %s → %s (%.1f%%)", m_name, result["prediction"], result["confidence"])
            model_results.append(result)
            if result["prediction"] == "Cataract":
                cataract_votes.append(result["confidence"])
            else:
                normal_votes.append(result["confidence"])

        cataract_count = len(cataract_votes)
        normal_count   = len(normal_votes)
        # Tie → Cataract (conservative for medical tool)
        final_pred     = "Cataract" if cataract_count >= normal_count else "Normal"
        winning_votes  = cataract_votes if final_pred == "Cataract" else normal_votes

        avg_conf    = sum(winning_votes) / len(winning_votes) if winning_votes else 0
        avg_entropy = sum(m["entropy"] for m in model_results) / len(model_results) if model_results else 0

        final_result = {
            "prediction":     final_pred,
            "confidence":     round(avg_conf, 2),
            "model_count":    len(model_results),
            "cataract_votes": cataract_count,
            "avg_entropy":    round(avg_entropy, 4),
        }

        if final_result["confidence"] < MIN_CONFIDENCE:
            return render_template("index.html", image_url=image_url,
                error_message=f"The model is not confident enough ({final_result['confidence']:.1f}%). "
                              "Please try a sharper, better-lit photo.")

        if final_result["avg_entropy"] > MAX_ENTROPY:
            return render_template("index.html", image_url=image_url,
                error_message="The AI models are uncertain about this image. "
                              "Please upload a clearer, well-focused close-up of the eye.")

        eye_features = analyze_eye_features(img_path)

        heatmap_url = None
        if model_results:
            best_m   = max(model_results, key=lambda x: x["confidence"])
            best_mdl = get_model(best_m["model"])
            if best_mdl:
                hm_path    = os.path.join(UPLOAD_FOLDER, f"hm_{safe_name}")
                target_cls = 0 if final_pred == "Cataract" else 1
                hm_result  = generate_gradcam(best_mdl, input_tensor, target_cls, img_path, hm_path)
                if hm_result:
                    heatmap_url = "/" + hm_path

        summary     = get_groq_summary(final_result, model_results, groq_api_key)
        explanation = get_cataract_explanation(final_pred)

        prediction_data = {
            "final":        final_result,
            "individual":   model_results,
            "summary":      summary,
            "eye_features": eye_features,
            "heatmap_url":  heatmap_url,
            "explanation":  explanation,
        }

        # Store only essential data in session (avoid 4KB cookie overflow)
        session["last_result"] = {
            "final":      final_result,
            "individual": model_results,
        }

    return render_template("index.html",
        prediction_data=prediction_data,
        image_url=image_url,
        error_message=error_message)


# ══════════════════════════════════════════════════════════════
#  CHAT ENDPOINT
# ══════════════════════════════════════════════════════════════
@app.route("/chat", methods=["POST"])
def chat():
    user_msg      = request.json.get("message", "").strip()
    selected_lang = request.json.get("language", "English")
    api_key       = request.json.get("api_key")
    last_result   = session.get("last_result")

    if not user_msg:
        return jsonify({"reply": "I did not receive your message. Please try again."})
    if not api_key:
        return jsonify({"reply": "API key is required for AI chat."})

    # Per-user chat memory in session (not global)
    chat_memory = session.get("chat_memory", [])
    if len(chat_memory) > 14:
        chat_memory = chat_memory[-14:]

    # Rate limit: max 50 messages per session
    chat_count = session.get("chat_count", 0)
    if chat_count >= 50:
        return jsonify({"reply": "Chat limit reached for this session. Please refresh to start a new session."})

    try:
        client  = Groq(api_key=api_key)
        context = (
            "The user just scanned their eye. "
            f"Result: {json.dumps(last_result['final'] if last_result else 'No scan yet')}."
        )
        system_prompt = textwrap.dedent(f"""
            You are a friendly AI Eye Assistant.
            Speak in plain everyday language.
            {context}
            Conversation memory (use for context): {chat_memory}
            Keep responses VERY BRIEF — 2 to 4 sentences max.
            RESPOND ONLY IN {selected_lang.upper()} LANGUAGE.
            IF TELUGU: use only Telugu script. NO English letters.
            IF HINDI: use only Devanagari script. NO English letters.
            NO asterisks (*) or square brackets ([]).
            Be professional but friendly.
        """).strip()

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.5, max_tokens=500,
        )
        reply = completion.choices[0].message.content
        chat_memory.append(f"User: {user_msg}")
        chat_memory.append(f"AI: {reply}")
        session["chat_memory"] = chat_memory
        session["chat_count"]  = chat_count + 1
        return jsonify({"reply": reply})

    except Exception as exc:
        return jsonify({"reply": f"Oops! I encountered an error: {str(exc)}"})


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)