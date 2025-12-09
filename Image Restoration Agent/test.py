# Install / update required libraries.
!pip uninstall -y google-generativeai -q      
!pip install -U -q google-genai piq scikit-image opencv-python-headless tqdm pandas

import os, cv2, json, torch
import numpy as np
import pandas as pd

from skimage import img_as_float
from scipy.signal import convolve2d
from scipy import special

from tqdm import tqdm
import piq

from google import genai
from google.genai import types  

from google.colab import drive
drive.mount('/content/drive')  

# Set your Google API key (replace with your own key, do NOT commit secrets to GitHub)
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# Create Gemini client
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Text+image planner model
GEMINI_PLANNER_MODEL = "gemini-2.5-flash"

# Image generation / editing model
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"

# Quick sanity check that the image model works
resp = client.models.generate_content(
    model=GEMINI_IMAGE_MODEL,
    contents="Generate a simple red circle",
    config={"response_modalities": ["IMAGE"]},
)
print("OK, image model works, parts:", len(resp.candidates[0].content.parts))


# ============= 1. Image I/O & Grayscale =============

def load_image_rgb(path):
    """Read image from disk and return RGB float32 in [0, 1]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return img_as_float(rgb).astype(np.float32)


def to_gray(rgb):
    """Convert RGB float32 [0, 1] to grayscale float32 [0, 1]."""
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


# ============= 2. NIQE proxy (MSCN + AGGD) =============

def compute_mscn(gray):
    """Compute MSCN coefficients for NIQE-like statistics."""
    kernel = np.array(
        [[1, 2, 3, 2, 1],
         [2, 4, 6, 4, 2],
         [3, 6, 8, 6, 3],
         [2, 4, 6, 4, 2],
         [1, 2, 3, 2, 1]],
        dtype=np.float32,
    )
    kernel /= kernel.sum()
    mu = convolve2d(gray, kernel, mode="same")
    sigma = np.sqrt(np.abs(convolve2d(gray * gray, kernel, mode="same") - mu * mu))
    return (gray - mu) / (sigma + 1e-6)


def estimate_aggd_beta(vec):
    """Estimate AGGD parameters used in NIQE-like computation."""
    gam = np.arange(0.2, 10, 0.001)
    r_gam = (special.gamma(2 / gam) ** 2) / (
        special.gamma(1 / gam) * special.gamma(3 / gam)
    )
    r_hat = (np.mean(np.abs(vec)) ** 2) / np.mean(vec ** 2)
    gamma = gam[np.argmin((r_gam - r_hat) ** 2)]

    sigma_l = np.sqrt(((vec[vec < 0]) ** 2).mean()) if np.any(vec < 0) else 0.0
    sigma_r = np.sqrt(((vec[vec > 0]) ** 2).mean()) if np.any(vec > 0) else 0.0
    return gamma, sigma_l, sigma_r


def compute_niqe(img_rgb):
    """Simplified NIQE proxy (higher is worse)."""
    gray = to_gray(img_rgb)
    mscn = compute_mscn(gray).flatten().astype(np.float32)
    if mscn.size == 0:
        return float("inf")
    gamma, sigma_l, sigma_r = estimate_aggd_beta(mscn)
    if gamma == 0:
        return float("inf")
    return float((sigma_l + sigma_r) / gamma)


# ============= 3. BRISQUE (via piq) =============

def compute_brisque_score(img_rgb):
    """Compute BRISQUE score with piq (lower is better)."""
    img_t = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float()
    try:
        with torch.no_grad():
            score = piq.brisque(img_t, data_range=1.0).item()
        return float(score)
    except Exception as e:
        print("BRISQUE failed:", e)
        return float("inf")


# ============= 4. IQA Features & Global Quality Score =============

def compute_basic_iqa_features(img_rgb):
    """Compute basic hand-crafted IQA features."""
    gray = to_gray(img_rgb)
    h, w = gray.shape[:2]

    brightness_mean = gray.mean()
    brightness_std = gray.std()
    p_dark = float(np.mean(gray < 30 / 255.0))
    p_bright = float(np.mean(gray > 225 / 255.0))

    lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
    lap_var = float(lap.var())

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_var = float((gray - blur).var())

    return {
        "height": float(h),
        "width": float(w),
        "brightness_mean": float(brightness_mean),
        "brightness_std": float(brightness_std),
        "p_dark": p_dark,
        "p_bright": p_bright,
        "contrast_std": float(brightness_std),
        "laplacian_var": lap_var,
        "noise_var": noise_var,
    }


def extract_iqa_features_from_rgb(img_rgb):
    """Compute basic features + NIQE + BRISQUE."""
    feats = compute_basic_iqa_features(img_rgb)
    feats["niqe"] = compute_niqe(img_rgb)
    feats["brisque"] = compute_brisque_score(img_rgb)
    return feats


def extract_iqa_features_from_path(path):
    """Convenience wrapper: load image then extract IQA features."""
    return extract_iqa_features_from_rgb(load_image_rgb(path))


def normalize_feature(x, min_val, max_val, invert=False):
    """Normalize scalar feature to [0, 1], with optional inversion."""
    x_norm = (x - min_val) / (max_val - min_val + 1e-8)
    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    return 1.0 - x_norm if invert else x_norm


def compute_quality_score(feats, w_contrast=0.4, w_sharpness=0.4, w_niqe=0.2):
    """
    Compute a single scalar quality score Q (higher is better):

        Q = w_contrast * norm(contrast) +
            w_sharpness * norm(sharpness) -
            w_niqe * norm(NIQE)
    """
    contrast = feats["contrast_std"]
    sharp = feats["laplacian_var"]
    niqe_val = feats["niqe"]

    c_n = normalize_feature(contrast, 0.0, 0.25)
    s_n = normalize_feature(sharp, 0.0, 300.0)
    n_n = normalize_feature(niqe_val, 0.0, 20.0)

    Q = w_contrast * c_n + w_sharpness * s_n - w_niqe * n_n
    return float(Q), {"contrast_norm": c_n, "sharpness_norm": s_n, "niqe_norm": n_n}


# ============= 5. Rule-based Diagnosis (SV/RS) =============

def diagnose_image_from_path(img_path, source_type=None):
    """
    Diagnose degradations from an image path.

    - Infer source_type (SV / RS) if not given.
    - Return: source_type, boolean problem flags, and IQA features.
    """
    feats = extract_iqa_features_from_path(img_path)
    full_lower = img_path.lower()
    ext = os.path.splitext(img_path)[1].lower()

    # Infer source type if not specified
    if source_type in ["SV", "RS"]:
        st = source_type
    elif "satellite" in full_lower or "rsi_" in full_lower:
        st = "RS"
    elif ext in [".tif", ".tiff"]:
        st = "RS"
    elif ext in [".jpg", ".jpeg", ".png"]:
        st = "SV"
    else:
        st = "SV"

    probs = {
        "low_light": False,
        "over_exposed": False,
        "low_contrast": False,
        "blur": False,
        "high_noise": False,
        "haze_like": False,
        "backlight": False,
        "low_resolution": False,
        "high_cloud": False,
    }

    f = feats

    # Global thresholds (slightly relaxed)
    if (f["brightness_mean"] < 0.5) or (f["p_dark"] > 0.2):
        probs["low_light"] = True
    if f["p_bright"] > 0.4 and f["brightness_mean"] > 0.7:
        probs["over_exposed"] = True
    if f["contrast_std"] < 0.05:
        probs["low_contrast"] = True
    if f["laplacian_var"] < 20:
        probs["blur"] = True
    if f["noise_var"] > 0.002:
        probs["high_noise"] = True
    if (
        0.3 < f["brightness_mean"] < 0.7
        and f["contrast_std"] < 0.06
        and f["niqe"] > 6
    ):
        probs["haze_like"] = True

    # Extra checks for SV
    if st == "SV":
        if f["brightness_mean"] < 0.55 and f["p_bright"] > 0.2:
            probs["backlight"] = True
    else:  # RS specific
        h, w = f["height"], f["width"]
        if min(h, w) < 512:
            probs["low_resolution"] = True
        if probs["haze_like"]:
            probs["high_cloud"] = True

    return {"source_type": st, "problems": probs, "features": feats}


# ========= 5bis. Decide whether restoration is needed =========

def should_restore(feats, probs, Q_before, q_thresh=0.35):
    """
    Decide if an image should go through the full restoration pipeline.
    Uses both global Q and rule-based problem flags.
    """
    if Q_before < q_thresh:
        return True

    problematic = [
        "low_light",
        "over_exposed",
        "low_contrast",
        "blur",
        "high_noise",
        "haze_like",
        "backlight",
        "low_resolution",
        "high_cloud",
    ]
    for k in problematic:
        if probs.get(k, False):
            return True
    return False


# ============= 6. Restoration Tools (SV + RS) =============

def SV_low_light_enhance(img_rgb):
    """SV: gamma + CLAHE enhancement for low-light / backlight scenes."""
    img = (img_rgb * 255).astype(np.uint8)
    gamma = 1.3
    img_gamma = np.power(np.clip(img / 255.0, 0, 1), 1.0 / gamma)
    img_gamma = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)

    ycrcb = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    out = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)
    return img_as_float(out).astype(np.float32)


def SV_deblur_unsharp(img_rgb):
    """SV: simple unsharp masking for deblurring."""
    img = (img_rgb * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    usm = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return np.clip(usm.astype(np.float32) / 255.0, 0.0, 1.0)


def RS_super_resolution_bicubic(img_rgb, scale=2):
    """RS: simple bicubic upsampling as super-resolution baseline."""
    h, w = img_rgb.shape[:2]
    img = (img_rgb * 255).astype(np.uint8)
    out = cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    return img_as_float(out).astype(np.float32)


def RS_dehaze_simple(img_rgb):
    """RS: CLAHE on Y channel + mild gamma correction."""
    img = (img_rgb * 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    out = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)
    out_f = img_as_float(out).astype(np.float32)
    gamma = 1.1
    out_gamma = np.power(np.clip(out_f, 0, 1), 1.0 / gamma)
    return np.clip(out_gamma, 0.0, 1.0)


TOOL_FUNCTIONS = {
    "SV_low_light_enhance": SV_low_light_enhance,
    "SV_deblur_unsharp": SV_deblur_unsharp,
    "RS_super_resolution_bicubic": RS_super_resolution_bicubic,
    "RS_dehaze_simple": RS_dehaze_simple,
}

ALLOWED_SV_TOOLS = ["SV_low_light_enhance", "SV_deblur_unsharp"]
ALLOWED_RS_TOOLS = ["RS_super_resolution_bicubic", "RS_dehaze_simple"]


def apply_tool_chain(img_rgb, tool_chain):
    """Apply a sequence of restoration tools in order."""
    out = img_rgb.copy()
    for name in tool_chain:
        func = TOOL_FUNCTIONS.get(name)
        if func is None:
            print(f"[WARN] Unknown tool: {name}")
            continue
        out = func(out)
    return out


# ============= 7. Baseline Q_base (defect-driven) =============

def estimate_defect_scores(feats):
    """
    Estimate per-dimension defect scores in [0, 1]:
    brightness / contrast / sharpness / noise (higher = worse).
    """
    b_mean = feats["brightness_mean"]
    contrast = feats["contrast_std"]
    sharp = feats["laplacian_var"]
    noise = feats["noise_var"]

    dark_score = max(0.0, 0.5 - b_mean) / 0.5
    bright_score = max(0.0, b_mean - 0.8) / 0.2
    brightness_defect = max(dark_score, bright_score)

    contrast_defect = max(0.0, 0.06 - contrast) / 0.06
    sharp_defect = max(0.0, 40.0 - sharp) / 40.0
    noise_defect = max(0.0, noise - 0.002) / 0.01

    return {
        "brightness": float(np.clip(brightness_defect, 0, 1)),
        "contrast": float(np.clip(contrast_defect, 0, 1)),
        "sharpness": float(np.clip(sharp_defect, 0, 1)),
        "noise": float(np.clip(noise_defect, 0, 1)),
    }


def choose_baseline_tools_from_defect(source_type, defect_scores):
    """
    Choose a single baseline tool from the worst defect dimension.
    Always select at least one tool to create a clear baseline.
    """
    worst_type = max(defect_scores, key=lambda k: defect_scores[k])
    if source_type == "SV":
        if worst_type in ["brightness", "contrast"]:
            tools = ["SV_low_light_enhance"]
        else:
            tools = ["SV_deblur_unsharp"]
    else:
        if worst_type in ["brightness", "contrast", "noise"]:
            tools = ["RS_dehaze_simple"]
        else:
            tools = ["RS_super_resolution_bicubic"]
    return tools, worst_type, defect_scores[worst_type]


def compute_Q_base_for_image(
    img_path, source_type_override=None, save_base_image=True, suffix="_base"
):
    """
    Compute Q_base (baseline quality) for a single image.

    Steps:
      1) Run rule-based diagnostics.
      2) Estimate defect scores per dimension.
      3) Choose one baseline tool.
      4) Apply baseline tool(s) and recompute Q.
      5) Optionally save baseline image.
    """
    diag = diagnose_image_from_path(img_path, source_type_override)
    source_type = diag["source_type"]
    feats_before = diag["features"]

    img_rgb = load_image_rgb(img_path)
    Q_before_base, _ = compute_quality_score(feats_before)

    defect_scores = estimate_defect_scores(feats_before)
    base_tools, worst_type, worst_score = choose_baseline_tools_from_defect(
        source_type, defect_scores
    )

    base_img = apply_tool_chain(img_rgb, base_tools)
    feats_after_base = extract_iqa_features_from_rgb(base_img)
    Q_base, _ = compute_quality_score(feats_after_base)

    base_img_path = None
    if save_base_image:
        folder = os.path.dirname(img_path)
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        base_img_path = os.path.join(folder, base_name + suffix + ext)
        out_bgr = cv2.cvtColor((base_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(base_img_path, out_bgr)

    return {
        "image": os.path.basename(img_path),
        "path": img_path,
        "source_type": source_type,
        "Q_before_base": float(Q_before_base),
        "Q_base": float(Q_base),
        "base_tools": base_tools,
        "base_image_path": base_img_path,
        "defect_scores": defect_scores,
        "worst_defect_type": worst_type,
        "worst_defect_score": worst_score,
    }


# ============= 8. Gemini Planner Prompt & Call =============

def encode_image_to_jpeg_bytes(img_rgb):
    """Encode RGB float32 image to JPEG bytes for Gemini Vision."""
    img_u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def build_gemini_planner_prompt(source_type, hazard_type, iqa_feats):
    """Build text prompt for the Gemini restoration planner."""
    allowed_tools = ALLOWED_SV_TOOLS if source_type == "SV" else ALLOWED_RS_TOOLS
    iqa_json = json.dumps(iqa_feats, indent=2)
    return f"""
You are an image restoration planner for disaster imagery.

Goal:
- Improve visual quality for downstream damage analysis.
- Use ONLY non-generative, deterministic tools.
- Decide what degradations are present and which tools to apply.

Image metadata:
- source_type = "{source_type}"
- hazard_type = "{hazard_type}"

No-reference IQA features:
{iqa_json}

Allowed tools (TOOL NAMES MUST MATCH EXACTLY):
{allowed_tools}

Possible problem types:
- "low_light", "backlight", "blur", "low_contrast",
- "noise", "low_resolution", "haze_or_cloud",
- "compression_artifacts", "none"

Return ONLY JSON:
{{
  "detected_problems": [...],
  "recommended_tools": [...],
  "agent_decision": "short English summary"
}}
"""


def normalize_tool_name(name, source_type):
    """Map model-predicted tool name to canonical tool name."""
    if not isinstance(name, str):
        return None
    s = name.strip()
    if not s:
        return None
    s_lower = s.lower()

    TOOL_SYNONYMS_SV = {
        "sv_low_light_enhance": "SV_low_light_enhance",
        "low_light_enhance": "SV_low_light_enhance",
        "low-light-enhance": "SV_low_light_enhance",
        "low light enhance": "SV_low_light_enhance",
        "sv_deblur_unsharp": "SV_deblur_unsharp",
        "deblur_unsharp": "SV_deblur_unsharp",
        "deblur": "SV_deblur_unsharp",
    }
    TOOL_SYNONYMS_RS = {
        "rs_super_resolution_bicubic": "RS_super_resolution_bicubic",
        "super_resolution_bicubic": "RS_super_resolution_bicubic",
        "super resolution": "RS_super_resolution_bicubic",
        "sr": "RS_super_resolution_bicubic",
        "rs_dehaze_simple": "RS_dehaze_simple",
        "dehaze_simple": "RS_dehaze_simple",
        "dehaze": "RS_dehaze_simple",
        "dehazing": "RS_dehaze_simple",
    }

    if source_type == "SV":
        if s in ALLOWED_SV_TOOLS:
            return s
        return TOOL_SYNONYMS_SV.get(s_lower)
    else:
        if s in ALLOWED_RS_TOOLS:
            return s
        return TOOL_SYNONYMS_RS.get(s_lower)


def tools_from_problems(detected_problems, source_type):
    """
    Rule-based fallback mapping from problem types to tool chain
    when the planner output is empty but problems are detected.
    """
    tools = []
    if source_type == "SV":
        if "low_light" in detected_problems or "backlight" in detected_problems:
            tools.append("SV_low_light_enhance")
        if "blur" in detected_problems:
            tools.append("SV_deblur_unsharp")
    else:
        if "low_resolution" in detected_problems:
            tools.append("RS_super_resolution_bicubic")
        if "haze_or_cloud" in detected_problems or "high_cloud" in detected_problems:
            tools.append("RS_dehaze_simple")
    out = []
    for t in tools:
        if t not in out:
            out.append(t)
    return out


def call_gemini_for_plan(img_rgb, source_type, hazard_type, iqa_feats):
    """Call Gemini planner model to get detected problems + tool chain."""
    prompt = build_gemini_planner_prompt(source_type, hazard_type, iqa_feats)
    img_bytes = encode_image_to_jpeg_bytes(img_rgb)

    resp = client.models.generate_content(
        model=GEMINI_PLANNER_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        ],
        config={"response_modalities": ["TEXT"]},
    )
    raw_text = resp.text.strip()

    # Strip possible ```json ... ``` wrappers
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        data = json.loads(raw_text)
    except Exception as e:
        print("[WARN] JSON parse failed:", e, "text:", raw_text[:200])
        data = {
            "detected_problems": ["none"],
            "recommended_tools": [],
            "agent_decision": "Gemini parsing failed; no restoration applied.",
        }

    detected_problems = data.get("detected_problems", [])
    raw_tools = data.get("recommended_tools", [])
    agent_decision = data.get("agent_decision", "")

    # Normalize tool names
    normalized = []
    for t in raw_tools:
        canon = normalize_tool_name(t, source_type)
        if canon and canon not in normalized:
            normalized.append(canon)

    # Filter by allowed tools
    allowed = ALLOWED_SV_TOOLS if source_type == "SV" else ALLOWED_RS_TOOLS
    normalized = [t for t in normalized if t in allowed]

    # Fallback mapping if there are problems but no tools
    if (not normalized) and detected_problems and detected_problems != ["none"]:
        fb = tools_from_problems(detected_problems, source_type)
        fb = [t for t in fb if t in allowed]
        if fb:
            normalized = fb
            agent_decision += " | Fallback tools: " + ", ".join(normalized)

    return detected_problems, normalized, agent_decision


# ============= 9. Merge Rule-based & Gemini Diagnostics =============

def reconcile_problems(rule_probs, detected_problems):
    """
    Merge rule-based and Gemini-based diagnostics into final_problems.

    Rules:
    - If Gemini returns ["none"], mark all problems as False.
    - Otherwise, start from rule_probs and force True for Gemini problems.
    """
    final = dict(rule_probs)
    gem2rule = {
        "low_light": "low_light",
        "backlight": "backlight",
        "blur": "blur",
        "low_contrast": "low_contrast",
        "noise": "high_noise",
        "low_resolution": "low_resolution",
        "haze_or_cloud": "haze_like",
        "compression_artifacts": None,
    }
    if (not detected_problems) or (detected_problems == ["none"]):
        for k in final:
            final[k] = False
        return final

    for p in detected_problems:
        key = gem2rule.get(p)
        if key and key in final:
            final[key] = True
    return final


# ============= 10. Full Gemini Agent Pipeline =============

def process_image_with_gemini_agent(
    img_path, source_type_override=None, hazard_type="unknown", delta_Q=0.01
):
    """
    Full planner pipeline for a single image:

      1) Rule-based diagnostic.
      2) Global quality Q_before.
      3) Decide if restoration is needed.
      4) If needed, call Gemini planner and apply tools.
      5) Measure Q_after and keep only if improvement >= delta_Q.
    """
    diag = diagnose_image_from_path(img_path, source_type_override)
    source_type = diag["source_type"]
    rule_probs = diag["problems"]
    feats_before = diag["features"]

    img_rgb = load_image_rgb(img_path)
    Q_before, _ = compute_quality_score(feats_before)

    # Early exit if restoration not needed
    need_restore = should_restore(feats_before, rule_probs, Q_before)
    if not need_restore:
        return {
            "image": os.path.basename(img_path),
            "path": img_path,
            "source_type": source_type,
            "hazard_type": hazard_type,
            "detected_problems": ["none"],
            "rule_problems": rule_probs,
            "final_problems": rule_probs,
            "recommended_tools": [],
            "tools_tried": [],
            "agent_decision": "Quality acceptable, no restoration needed.",
            "accepted": 0,
            "Q_before": float(Q_before),
            "Q_after": float(Q_before),
            "delta_Q": 0.0,
            "features_before": feats_before,
            "features_after": feats_before,
        }

    detected_problems, tool_chain, agent_decision = call_gemini_for_plan(
        img_rgb, source_type, hazard_type, feats_before
    )
    final_probs = reconcile_problems(rule_probs, detected_problems)
    tools_tried = list(tool_chain)

    if not tool_chain:
        # Planner decided "do nothing" or produced invalid tools
        return {
            "image": os.path.basename(img_path),
            "path": img_path,
            "source_type": source_type,
            "hazard_type": hazard_type,
            "detected_problems": detected_problems,
            "rule_problems": rule_probs,
            "final_problems": final_probs,
            "recommended_tools": [],
            "tools_tried": tools_tried,
            "agent_decision": agent_decision,
            "accepted": 0,
            "Q_before": float(Q_before),
            "Q_after": float(Q_before),
            "delta_Q": 0.0,
            "features_before": feats_before,
            "features_after": feats_before,
        }

    # Apply tool chain
    img_after = apply_tool_chain(img_rgb, tool_chain)
    feats_after = extract_iqa_features_from_rgb(img_after)
    Q_after, _ = compute_quality_score(feats_after)

    # Keep only if quality gain is large enough
    if Q_after >= Q_before + delta_Q:
        accepted = 1
        tools_final = tools_tried
    else:
        accepted = 0
        img_after = img_rgb
        feats_after = feats_before
        Q_after = Q_before
        tools_final = []

    return {
        "image": os.path.basename(img_path),
        "path": img_path,
        "source_type": source_type,
        "hazard_type": hazard_type,
        "detected_problems": detected_problems,
        "rule_problems": rule_probs,
        "final_problems": final_probs,
        "recommended_tools": tools_final,
        "tools_tried": tools_tried,
        "agent_decision": agent_decision,
        "accepted": int(accepted),
        "Q_before": float(Q_before),
        "Q_after": float(Q_after),
        "delta_Q": float(Q_after - Q_before),
        "features_before": feats_before,
        "features_after": feats_after,
    }


# ============= 11. Gemini Image-only (“Nano”) Branch =============

def enhance_image_with_gemini_image_only(img_rgb, source_type, hazard_type="unknown"):
    """
    Call GEMINI_IMAGE_MODEL to directly enhance an image
    (brightness / contrast / sharpness / noise),
    while preserving real structures and damage.
    """
    img_bytes = encode_image_to_jpeg_bytes(img_rgb)

    prompt = f"""
You are a restoration expert for disaster imagery.

Image metadata:
- source_type = "{source_type}" (SV street-view, RS remote sensing)
- hazard_type = "{hazard_type}"

Task:
- Enhance brightness, contrast, and sharpness if necessary.
- Reduce noise and haze.
- Preserve true damage patterns and structures.
- Do NOT hallucinate new buildings or remove existing damage.
- Return ONE enhanced version of the same scene.
"""

    try:
        resp = client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            ],
            config={"response_modalities": ["IMAGE"]},
        )

        for part in resp.candidates[0].content.parts:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue
            data = inline.data  # raw image bytes
            nparr = np.frombuffer(data, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return img_as_float(rgb).astype(np.float32)

        print("[WARN] image model returned no inline_data, using original.")
        return img_rgb

    except Exception as e:
        print("[ERROR] enhance_image_with_gemini_image_only:", e)
        return img_rgb


def process_image_with_gemini_image_only(
    img_path,
    source_type_override=None,
    hazard_type="unknown",
    save_nano_image=True,
    suffix="_nano",
    delta_Q_accept=0.0,
):
    """
    Nano branch for a single image:

      1) Run rule-based diagnostic and compute Q_before.
      2) Use GEMINI_IMAGE_MODEL for end-to-end enhancement.
      3) Recompute IQA and Q_nano.
      4) Optionally save enhanced image and mark whether Q improved.
    """
    diag = diagnose_image_from_path(img_path, source_type_override)
    source_type = diag["source_type"]
    feats_before = diag["features"]

    img_rgb = load_image_rgb(img_path)
    Q_before, _ = compute_quality_score(feats_before)

    nano_ok = 1
    try:
        img_nano = enhance_image_with_gemini_image_only(
            img_rgb, source_type=source_type, hazard_type=hazard_type
        )
        if img_nano is None:
            raise RuntimeError("enhance_image_with_gemini_image_only returned None")
        img_nano = np.clip(img_nano.astype(np.float32), 0.0, 1.0)
    except Exception as e:
        print(f"[Nano] failed for {img_path}:", e)
        img_nano = img_rgb.copy()
        nano_ok = 0

    feats_nano = extract_iqa_features_from_rgb(img_nano)
    Q_nano, _ = compute_quality_score(feats_nano)
    accepted_nano = int(Q_nano >= Q_before + delta_Q_accept)

    nano_image_path = None
    if save_nano_image:
        folder = os.path.dirname(img_path)
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        nano_image_path = os.path.join(folder, base_name + suffix + ext)
        out_bgr = cv2.cvtColor(
            (np.clip(img_nano, 0, 1) * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(nano_image_path, out_bgr)

    return {
        "image": os.path.basename(img_path),
        "path": img_path,
        "source_type": source_type,
        "hazard_type": hazard_type,
        "Q_before": float(Q_before),
        "Q_nano": float(Q_nano),
        "features_before": feats_before,
        "features_nano": feats_nano,
        "nano_image_path": nano_image_path,
        "nano_ok": int(nano_ok),
        "accepted_nano": int(accepted_nano),
    }


# ============= 12. Fast Pre-screening: Need Restoration? =============

def need_restoration_basic(
    img_path,
    source_type_override=None,
    thr_brightness_low=0.35,
    thr_brightness_high=0.80,
    thr_contrast=0.04,
    thr_lap=15.0,
):
    """
    Very fast screening:
    Use only simple statistics (brightness / contrast / Laplacian)
    to decide whether an image should enter the full restoration branch.

    Returns:
      need_restore : bool   -> True if we run Q_base / Q_gemini / Q_nano
      quick_feats  : dict   -> brightness_mean / contrast_std / laplacian_var
      reason       : str    -> short text explanation
    """
    img_rgb = load_image_rgb(img_path)
    gray = to_gray(img_rgb)

    brightness_mean = float(gray.mean())
    contrast_std = float(gray.std())
    lap_var = float(
        cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F).var()
    )

    quick_feats = {
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "laplacian_var": lap_var,
    }

    reasons = []

    # Too dark / too bright
    if brightness_mean < thr_brightness_low:
        reasons.append("too dark")
    elif brightness_mean > thr_brightness_high:
        reasons.append("too bright")

    # Too low contrast
    if contrast_std < thr_contrast:
        reasons.append("low contrast")

    # Too low sharpness
    if lap_var < thr_lap:
        reasons.append("blurry")

    if len(reasons) == 0:
        need_restore = False
        reason_text = (
            "No restoration needed (brightness/contrast/sharpness within normal range)."
        )
    else:
        need_restore = True
        reason_text = "Need restoration: " + ", ".join(reasons)

    return need_restore, quick_feats, reason_text


# ============= 13. Batch Processing (Fast Version) =============

def batch_process_with_gemini_agent(
    root_folder,
    csv_out_path,
    default_hazard="unknown",
    delta_Q=0.01,
    save_samples_limit=20,
):
    """
    Fast batch pipeline:

      1) For each image, run need_restoration_basic.
         - If "no restoration needed":
             accepted = 0
             Only write one CSV row, Q_base / Q_gemini / Q_nano = NaN.
         - If "restoration needed":
             accepted = 1
             Run full Q_base / Q_gemini / Q_nano pipeline.

      2) Only save the first `save_samples_limit` accepted images as visual samples:
         - Original image stays in the original folder.
         - In `samples_dir`, save 3 restored versions:
             * baseline (Q_base)
             * Gemini planner (Q_gemini)
             * Nano (GEMINI_IMAGE_MODEL, Q_nano)

    CSV columns:
      image, detected_problems, agent_decision,
      Q_before, Q_base, Q_gemini, Q_nano, accepted
    """
    rows = []
    samples_saved = 0

    # Folder to store a small number of sample restoration images
    samples_dir = os.path.join(root_folder, f"_samples_{save_samples_limit}")
    os.makedirs(samples_dir, exist_ok=True)

    print("Scanning:", root_folder)

    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Skip sample subfolders to avoid processing generated results again
        dirnames[:] = [d for d in dirnames if not d.startswith("_samples_")]

        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()

            # Only process image files
            if ext not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                continue

            # Skip label TIFFs
            if fname.lower().startswith("label") and ext in [".tif", ".tiff"]:
                print("[Skip label]:", fpath)
                continue

            # Skip previously generated restoration results
            lower_name = fname.lower()
            if any(
                suffix in lower_name for suffix in ["_base", "_nano", "_gemini", "_src"]
            ):
                continue

            try:
                hazard_type = default_hazard

                # (1) Fast screening: need restoration?
                need_restore, quick_feats, quick_reason = need_restoration_basic(fpath)

                if not need_restore:
                    # For "no restoration needed" images:
                    # - accepted = 0
                    # - Q_base / Q_gemini / Q_nano = NaN
                    # - Approximate Q_before using quick_feats + fake NIQE
                    fake_feats = dict(quick_feats)
                    fake_feats["niqe"] = 6.0  # a mid-level NIQE just for Q estimation
                    Q_before_fast, _ = compute_quality_score(fake_feats)

                    row = {
                        "image": fname,
                        "detected_problems": "",
                        "agent_decision": quick_reason,
                        "Q_before": float(Q_before_fast),
                        "Q_base": math.nan,
                        "Q_gemini": math.nan,
                        "Q_nano": math.nan,
                        "accepted": 0,
                    }
                    rows.append(row)
                    continue

                # (2) Run full restoration pipeline for images that need it

                # 2.1 baseline
                base_res = compute_Q_base_for_image(
                    fpath,
                    source_type_override=None,
                    save_base_image=False,
                    suffix="_base",
                )

                # 2.2 Gemini planner + rule-based tools
                res = process_image_with_gemini_agent(
                    fpath,
                    hazard_type=hazard_type,
                    delta_Q=delta_Q,
                )

                # 2.3 Nano image-only enhancement
                nano_res = process_image_with_gemini_image_only(
                    fpath,
                    source_type_override=None,
                    hazard_type=hazard_type,
                    save_nano_image=False,
                    suffix="_nano",
                )

                # (3) Save sample images (only first N)
                if samples_saved < save_samples_limit:
                    img_rgb = load_image_rgb(fpath)

                    # Baseline image
                    base_tools = base_res["base_tools"]
                    img_base = apply_tool_chain(img_rgb, base_tools)

                    # Gemini planner image (if tools exist)
                    if res["recommended_tools"]:
                        img_gem = apply_tool_chain(img_rgb, res["recommended_tools"])
                    else:
                        img_gem = img_rgb.copy()

                    # Nano image (call image-only branch again)
                    img_nano = enhance_image_with_gemini_image_only(
                        img_rgb,
                        source_type=res["source_type"],
                        hazard_type=hazard_type,
                    )

                    # Preserve subfolder hierarchy inside samples_dir
                    rel_path = os.path.relpath(fpath, root_folder)
                    rel_dir = os.path.dirname(rel_path)
                    base_name, ext2 = os.path.splitext(os.path.basename(rel_path))

                    out_dir = os.path.join(samples_dir, rel_dir)
                    os.makedirs(out_dir, exist_ok=True)

                    # Baseline
                    out_base = os.path.join(out_dir, base_name + "_base" + ext2)
                    out_bgr_base = cv2.cvtColor(
                        (np.clip(img_base, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    )
                    cv2.imwrite(out_base, out_bgr_base)

                    # Gemini planner
                    out_gem = os.path.join(out_dir, base_name + "_gemini" + ext2)
                    out_bgr_gem = cv2.cvtColor(
                        (np.clip(img_gem, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    )
                    cv2.imwrite(out_gem, out_bgr_gem)

                    # Nano
                    out_nano = os.path.join(out_dir, base_name + "_nano" + ext2)
                    out_bgr_nano = cv2.cvtColor(
                        (np.clip(img_nano, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    )
                    cv2.imwrite(out_nano, out_bgr_nano)

                    samples_saved += 1

                # (4) Write CSV row for images that went through restoration
                row = {
                    "image": res["image"],
                    "detected_problems": ";".join(res["detected_problems"])
                    if res["detected_problems"]
                    else "",
                    "agent_decision": res["agent_decision"],
                    "Q_before": float(res["Q_before"]),
                    "Q_base": float(base_res["Q_base"]),
                    "Q_gemini": float(res["Q_after"]),
                    "Q_nano": float(nano_res["Q_nano"]),
                    "accepted": 1,
                }
                rows.append(row)

            except Exception as e:
                print(f"[Error] {fpath}: {e}")
                continue

    # ============= Save results to CSV =============
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_out_path), exist_ok=True)
    df.to_csv(csv_out_path, index=False, encoding="utf-8-sig")

    print("Saved CSV:", csv_out_path)
    print("Total images:", len(rows))
    print("Accepted (need restoration):", int((df["accepted"] == 1).sum()))
    print("Samples saved (images with 3 restored copies):", samples_saved)

    return df


# ============= 14. Run Batch Pipeline =============

root_folder = "/content/drive/MyDrive/25Fall/25Fall class/generative ai/Dataset/SVI_IncidentsDataset"
csv_out_path = os.path.join(root_folder, "agent_results_gemini3_image.csv")

df_results = batch_process_with_gemini_agent(
    root_folder=root_folder,
    csv_out_path=csv_out_path,
    default_hazard="hurricane",
    delta_Q=0.01,
    save_samples_limit=30,
)

df_results.head()
