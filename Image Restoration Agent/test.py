from google.colab import drive
drive.mount('/content/drive')

# Colab-only dependency installation
!pip install -q scikit-image piq opencv-python-headless tqdm pandas google-generativeai

import os
import cv2
import numpy as np
from skimage import img_as_float
from scipy.signal import convolve2d
from scipy import special
import torch
import piq
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
import json

# =========================
# Gemini configuration
# =========================
# Expect GEMINI_API_KEY to be set in the environment externally
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

GEMINI_MODEL_NAME = "gemini-2.5-flash"


# =========================
# 1. Image I/O and grayscale conversion
# =========================

def load_image_rgb(path):
    """Load an image as RGB float32 in [0, 1]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = img_as_float(rgb)
    return rgb


def to_gray(rgb):
    """Convert RGB float32 [0, 1] to grayscale float32 [0, 1]."""
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


# =========================
# 2. NIQE proxy (MSCN + AGGD)
# =========================

def compute_mscn(gray):
    """Compute MSCN coefficients for a grayscale image."""
    kernel = np.array(
        [
            [1, 2, 3, 2, 1],
            [2, 4, 6, 4, 2],
            [3, 6, 8, 6, 3],
            [2, 4, 6, 4, 2],
            [1, 2, 3, 2, 1],
        ],
        dtype=np.float32,
    )
    kernel = kernel / kernel.sum()
    mu = convolve2d(gray, kernel, mode="same")
    sigma = np.sqrt(np.abs(convolve2d(gray * gray, kernel, mode="same") - mu * mu))
    mscn = (gray - mu) / (sigma + 1e-6)
    return mscn


def estimate_aggd_beta(vec):
    """Estimate AGGD parameters used in NIQE-like computation."""
    gam = np.arange(0.2, 10, 0.001)
    r_gam = (special.gamma(2 / gam) ** 2) / (
        special.gamma(1 / gam) * special.gamma(3 / gam)
    )
    r_hat = (np.mean(np.abs(vec))) ** 2 / np.mean(vec**2)
    gamma = gam[np.argmin((r_gam - r_hat) ** 2)]

    sigma_l = np.sqrt(((vec[vec < 0]) ** 2).mean()) if np.any(vec < 0) else 0.0
    sigma_r = np.sqrt(((vec[vec > 0]) ** 2).mean()) if np.any(vec > 0) else 0.0
    return gamma, sigma_l, sigma_r


def compute_niqe(img_rgb):
    """Simplified NIQE proxy (higher is worse)."""
    gray = to_gray(img_rgb)
    mscn = compute_mscn(gray)
    vec = mscn.flatten().astype(np.float32)
    if vec.size == 0:
        return float("inf")
    gamma, sigma_l, sigma_r = estimate_aggd_beta(vec)
    if gamma == 0:
        return float("inf")
    return float((sigma_l + sigma_r) / gamma)


# =========================
# 3. BRISQUE (via piq)
# =========================

def compute_brisque_score(img_rgb):
    """Compute BRISQUE score with piq. Lower is better."""
    img_t = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float()
    try:
        with torch.no_grad():
            score = piq.brisque(img_t, data_range=1.0).item()
        return float(score)
    except Exception as e:
        print("BRISQUE evaluation failed:", e)
        return float("inf")


# =========================
# 4. Basic IQA features
# =========================

def compute_basic_iqa_features(img_rgb):
    """Compute basic no-reference IQA features."""
    gray = to_gray(img_rgb)
    h, w = gray.shape[:2]

    brightness_mean = gray.mean()
    brightness_std = gray.std()
    p_dark = float(np.mean(gray < 30 / 255.0))
    p_bright = float(np.mean(gray > 225 / 255.0))

    contrast_std = brightness_std

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
        "contrast_std": float(contrast_std),
        "laplacian_var": lap_var,
        "noise_var": noise_var,
    }


def extract_iqa_features_from_rgb(img_rgb):
    """Compute all IQA features (basic + NIQE + BRISQUE) from an RGB image."""
    feats = compute_basic_iqa_features(img_rgb)
    feats["niqe"] = compute_niqe(img_rgb)
    feats["brisque"] = compute_brisque_score(img_rgb)
    return feats


def extract_iqa_features_from_path(img_path):
    """Load an image from path and compute IQA features."""
    img_rgb = load_image_rgb(img_path)
    return extract_iqa_features_from_rgb(img_rgb)


# =========================
# 5. Combined quality score Q
# =========================

def normalize_feature(x, min_val, max_val, invert=False):
    """Normalize a scalar feature to [0, 1], optionally inverted."""
    x_norm = (x - min_val) / (max_val - min_val + 1e-8)
    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    if invert:
        x_norm = 1.0 - x_norm
    return x_norm


def compute_quality_score(
    feats, w_contrast=0.4, w_sharpness=0.4, w_niqe=0.2
):
    """
    Compute a scalar quality score:
        Q = w_contrast * norm(contrast) +
            w_sharpness * norm(sharpness) -
            w_niqe * norm(NIQE)
    Higher Q is better.
    """
    contrast = feats["contrast_std"]
    sharpness = feats["laplacian_var"]
    niqe_val = feats["niqe"]

    contrast_norm = normalize_feature(contrast, 0.0, 0.25, invert=False)
    sharpness_norm = normalize_feature(sharpness, 0.0, 300.0, invert=False)
    niqe_norm = normalize_feature(niqe_val, 0.0, 20.0, invert=False)

    Q = (
        w_contrast * contrast_norm
        + w_sharpness * sharpness_norm
        - w_niqe * niqe_norm
    )

    return float(Q), {
        "contrast_norm": contrast_norm,
        "sharpness_norm": sharpness_norm,
        "niqe_norm": niqe_norm,
    }


# =========================
# 6. Rule-based diagnostic (SV/RS)
# =========================

def diagnose_image_from_path(img_path, source_type=None):
    """
    Diagnose degradations from an image path.

    - Infers source_type (SV/RS) from path/extension if not provided.
    - Returns source_type, a problem flag vector, and IQA features.
    """
    feats = extract_iqa_features_from_path(img_path)
    full_path_lower = img_path.lower()
    ext = os.path.splitext(img_path)[1].lower()

    # Decide source_type
    if source_type in ["SV", "RS"]:
        st = source_type
    elif "satellite" in full_path_lower or "rsi_" in full_path_lower:
        st = "RS"
    elif ext in [".tif", ".tiff"]:
        st = "RS"
    elif ext in [".jpg", ".jpeg", ".png"]:
        st = "SV"
    else:
        st = "SV"

    # Initialize problem flags
    probs = {
        "low_light": False,
        "over_exposed": False,
        "low_contrast": False,
        "blur": False,
        "high_noise": False,
        "haze_like": False,
        "backlight": False,       # SV only
        "low_resolution": False,  # RS only
        "high_cloud": False,      # RS only
    }

    f = feats

    # Generic IQA-based rules (slightly relaxed)
    # Low-light (either dark mean or large dark area)
    if (f["brightness_mean"] < 0.5) or (f["p_dark"] > 0.2):
        probs["low_light"] = True

    # Over-exposure
    if f["p_bright"] > 0.4 and f["brightness_mean"] > 0.7:
        probs["over_exposed"] = True

    # Low contrast
    if f["contrast_std"] < 0.05:
        probs["low_contrast"] = True

    # Blur
    if f["laplacian_var"] < 20:
        probs["blur"] = True

    # Noise
    if f["noise_var"] > 0.002:
        probs["high_noise"] = True

    # Haze/cloud-like: mid brightness + low contrast + high NIQE
    if (
        0.3 < f["brightness_mean"] < 0.7
        and f["contrast_std"] < 0.06
        and f["niqe"] > 6
    ):
        probs["haze_like"] = True

    # SV: backlight
    if st == "SV":
        if (f["brightness_mean"] < 0.55) and (f["p_bright"] > 0.2):
            probs["backlight"] = True

    # RS: resolution + cloud
    if st == "RS":
        h = f["height"]
        w = f["width"]
        if min(h, w) < 512:
            probs["low_resolution"] = True
        if probs["haze_like"]:
            probs["high_cloud"] = True

    return {"source_type": st, "problems": probs, "features": feats}


# =========================
# 7. Restoration tools (SV + RS)
# =========================

def SV_low_light_enhance(img_rgb):
    """SV: gamma + CLAHE-based enhancement for low-light/backlight scenes."""
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
    return out.astype(np.float32) / 255.0


def SV_deblur_unsharp(img_rgb):
    """SV: simple unsharp masking for deblurring."""
    img = (img_rgb * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    usm = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return np.clip(usm.astype(np.float32) / 255.0, 0.0, 1.0)


def RS_super_resolution_bicubic(img_rgb, scale=2):
    """RS: placeholder SR via bicubic upsampling."""
    h, w = img_rgb.shape[:2]
    img = (img_rgb * 255).astype(np.uint8)
    out = cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    return out.astype(np.float32) / 255.0


def RS_dehaze_simple(img_rgb):
    """RS: simple dehazing via CLAHE on Y channel + mild gamma correction."""
    img = (img_rgb * 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    out = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)
    out_f = out.astype(np.float32) / 255.0
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
        func = TOOL_FUNCTIONS.get(name, None)
        if func is None:
            print(f"[WARN] Unknown tool: {name}, skip.")
            continue
        out = func(out)
    return out


# =========================
# 8. Tool name normalization and rule fallback
# =========================

# Synonyms → canonical tool names
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


def normalize_tool_name(name, source_type):
    """Map a model-predicted tool name to a canonical tool name."""
    if not isinstance(name, str):
        return None
    s = name.strip()
    if not s:
        return None
    s_lower = s.lower()

    if source_type == "SV":
        if s in ALLOWED_SV_TOOLS:
            return s
        return TOOL_SYNONYMS_SV.get(s_lower, None)
    else:
        if s in ALLOWED_RS_TOOLS:
            return s
        return TOOL_SYNONYMS_RS.get(s_lower, None)


def tools_from_problems(detected_problems, source_type):
    """
    Rule-based fallback mapping from problem types to a tool chain.
    Used when the planner fails to produce a valid tool list.
    """
    tools = []
    if source_type == "SV":
        if "low_light" in detected_problems or "backlight" in detected_problems:
            tools.append("SV_low_light_enhance")
        if "blur" in detected_problems:
            tools.append("SV_deblur_unsharp")
    else:  # RS
        if "low_resolution" in detected_problems:
            tools.append("RS_super_resolution_bicubic")
        if "haze_or_cloud" in detected_problems or "high_cloud" in detected_problems:
            tools.append("RS_dehaze_simple")

    # Deduplicate while preserving order
    tools_unique = []
    for t in tools:
        if t not in tools_unique:
            tools_unique.append(t)
    return tools_unique


# =========================
# 9. Gemini Vision Planner
# =========================

def encode_image_to_jpeg_bytes(img_rgb):
    """Encode an RGB image to JPEG bytes for Gemini Vision."""
    img_u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def build_gemini_planner_prompt(source_type, hazard_type, iqa_feats):
    """Build the text prompt for the Gemini restoration planner."""
    allowed_tools = ALLOWED_SV_TOOLS if source_type == "SV" else ALLOWED_RS_TOOLS
    iqa_json = json.dumps(iqa_feats, indent=2)

    prompt = f"""
You are an image restoration planner for disaster imagery.

Goal:
- Improve visual quality for downstream damage analysis.
- Use ONLY non-generative, deterministic tools.
- Decide what degradations are present and which tools to apply.

Image metadata:
- source_type = "{source_type}"  (SV = street-view, RS = remote sensing)
- hazard_type = "{hazard_type}"

No-reference IQA features:
{iqa_json}

Allowed tools for this image (TOOL NAMES MUST MATCH EXACTLY):
{allowed_tools}

Examples of valid tool lists:
- ["SV_low_light_enhance"]
- ["SV_low_light_enhance", "SV_deblur_unsharp"]
- ["RS_super_resolution_bicubic"]
- ["RS_dehaze_simple"]
- []

Possible problem types:
- "low_light"
- "backlight"
- "blur"
- "low_contrast"
- "noise"
- "low_resolution"
- "haze_or_cloud"
- "compression_artifacts"
- "none"

Your response must be a pure JSON object with fields:
{{
  "detected_problems": ["low_light", "blur", ...],
  "recommended_tools": ["SV_low_light_enhance", "SV_deblur_unsharp"],
  "agent_decision": "short English sentence summarizing problem and action"
}}

Constraints:
- "recommended_tools" MUST be a subset (or empty) of the allowed tools above.
- Use the tool names EXACTLY as given (case-sensitive).
- If quality is already good, use "detected_problems": ["none"] and [] for "recommended_tools".
- Do not output any text outside the JSON (no explanation lines).
"""
    return prompt


def call_gemini_for_plan(img_rgb, source_type, hazard_type, iqa_feats):
    """Call Gemini Vision to get a restoration plan."""
    prompt = build_gemini_planner_prompt(source_type, hazard_type, iqa_feats)
    img_bytes = encode_image_to_jpeg_bytes(img_rgb)

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    response = model.generate_content(
        [prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
    )

    raw_text = response.text.strip()

    # Strip possible ```json ... ``` wrappers
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        data = json.loads(raw_text)
    except Exception as e:
        print("[WARN] JSON parse failed, fallback to no-tools:", e)
        data = {
            "detected_problems": ["none"],
            "recommended_tools": [],
            "agent_decision": "Gemini parsing failed; no restoration applied.",
        }

    detected_problems = data.get("detected_problems", [])
    raw_recommended = data.get("recommended_tools", [])
    agent_decision = data.get("agent_decision", "")

    # Normalize tool names
    normalized_tools = []
    for t in raw_recommended:
        canon = normalize_tool_name(t, source_type)
        if canon and canon not in normalized_tools:
            normalized_tools.append(canon)

    # Filter with allowed list (extra safety)
    allowed = ALLOWED_SV_TOOLS if source_type == "SV" else ALLOWED_RS_TOOLS
    normalized_tools = [t for t in normalized_tools if t in allowed]

    # Fallback: if tools are empty but problems are non-trivial, infer from rules
    if len(normalized_tools) == 0 and detected_problems and detected_problems != [
        "none"
    ]:
        fallback = tools_from_problems(detected_problems, source_type)
        fallback = [t for t in fallback if t in allowed]
        if fallback:
            normalized_tools = fallback
            agent_decision = (
                agent_decision
                + " | Fallback: tools inferred from detected problems: "
                + ", ".join(normalized_tools)
            )

    return detected_problems, normalized_tools, agent_decision


# =========================
# 10. Gemini Agent pipeline
# =========================

def reconcile_problems(rule_probs, detected_problems):
    """
    Merge rule-based and Gemini-based diagnostics into final_problems.

    Rules:
    - If Gemini returns ["none"], all problems are set to False.
    - Otherwise, start from rule_probs and force True for any Gemini problem.
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
        for k in final.keys():
            final[k] = False
        return final

    for p in detected_problems:
        key = gem2rule.get(p)
        if key is not None and key in final:
            final[key] = True

    return final


def process_image_with_gemini_agent(
    img_path, source_type_override=None, hazard_type="unknown", delta_Q=0.01
):
    """
    Full pipeline for a single image:

      1) Rule-based diagnostic (SV/RS + IQA + rule_problems)
      2) Gemini planner:
         - detected_problems
         - recommended_tools
         - agent_decision
      3) Merge diagnostics into final_problems (Gemini-prioritized)
      4) Apply tool chain, compute Q_before / Q_after
      5) Verify gain and optionally rollback
    """
    # 1) Rule-based diagnostic
    diag = diagnose_image_from_path(img_path, source_type=source_type_override)
    source_type = diag["source_type"]
    rule_probs = diag["problems"]
    feats_before = diag["features"]

    img_rgb = load_image_rgb(img_path)
    Q_before, _ = compute_quality_score(feats_before)

    # 2) Gemini planner
    detected_problems, tool_chain, agent_decision = call_gemini_for_plan(
        img_rgb=img_rgb,
        source_type=source_type,
        hazard_type=hazard_type,
        iqa_feats=feats_before,
    )

    # 3) Merge diagnostics
    final_probs = reconcile_problems(rule_probs, detected_problems)

    tools_tried = list(tool_chain)

    # No tools → return early
    if len(tool_chain) == 0:
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

    # 4) Apply restoration tools
    img_after = apply_tool_chain(img_rgb, tool_chain)

    # 5) Recompute IQA and quality score
    feats_after = extract_iqa_features_from_rgb(img_after)
    Q_after, _ = compute_quality_score(feats_after)

    # 6) Verification / rollback
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


# =========================
# 11. Q_base baseline: “worst-dimension” heuristic
# =========================

def estimate_defect_scores(feats):
    """
    Estimate rough defect scores in [0, 1] across dimensions:

      - brightness: too dark or too bright
      - contrast  : low contrast
      - sharpness : blur (low Laplacian variance)
      - noise     : high noise variance

    Higher score indicates worse quality for that dimension.
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
        "brightness": float(np.clip(brightness_defect, 0.0, 1.0)),
        "contrast": float(np.clip(contrast_defect, 0.0, 1.0)),
        "sharpness": float(np.clip(sharp_defect, 0.0, 1.0)),
        "noise": float(np.clip(noise_defect, 0.0, 1.0)),
    }


def choose_baseline_tools_from_defect(source_type, defect_scores):
    """
    Choose a single baseline tool from the worst defect dimension.

    This intentionally always selects at least one tool (no thresholds),
    so the baseline is guaranteed to perform some modification, providing
    a clear reference against the Gemini-based agent.
    """
    worst_type = max(defect_scores.keys(), key=lambda k: defect_scores[k])
    worst_score = defect_scores[worst_type]

    if source_type == "SV":
        if worst_type in ["brightness", "contrast"]:
            tool_chain = ["SV_low_light_enhance"]
        elif worst_type == "sharpness":
            tool_chain = ["SV_deblur_unsharp"]
        elif worst_type == "noise"]:
            tool_chain = ["SV_deblur_unsharp"]
        else:
            tool_chain = ["SV_low_light_enhance"]
    else:  # RS
        if worst_type in ["brightness", "contrast", "noise"]:
            tool_chain = ["RS_dehaze_simple"]
        elif worst_type == "sharpness":
            tool_chain = ["RS_super_resolution_bicubic"]
        else:
            tool_chain = ["RS_dehaze_simple"]

    return tool_chain, worst_type, worst_score


def compute_Q_base_for_image(
    img_path, source_type_override=None, save_base_image=True, suffix="_base"
):
    """
    Compute Q_base (baseline quality) for a single image:

      1) Run rule-based diagnostic to get features + source_type.
      2) Compute defect_scores per dimension.
      3) Choose a single baseline tool based on the worst dimension.
      4) Apply the baseline tool chain.
      5) Compute baseline quality Q_base.
      6) Optionally save the baseline image with suffix (e.g., *_base.png).
    """
    diag = diagnose_image_from_path(img_path, source_type=source_type_override)
    source_type = diag["source_type"]
    feats_before = diag["features"]

    img_rgb = load_image_rgb(img_path)
    Q_before_base, _ = compute_quality_score(feats_before)

    defect_scores = estimate_defect_scores(feats_before)

    base_tools, worst_defect_type, worst_defect_score = (
        choose_baseline_tools_from_defect(
            source_type=source_type, defect_scores=defect_scores
        )
    )

    base_img = apply_tool_chain(img_rgb, base_tools)

    feats_after_base = extract_iqa_features_from_rgb(base_img)
    Q_base, _ = compute_quality_score(feats_after_base)

    base_img_path = None
    if save_base_image:
        folder = os.path.dirname(img_path)
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        base_img_path = os.path.join(folder, base_name + suffix + ext)

        out_bgr = cv2.cvtColor(
            (base_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
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
        "worst_defect_type": worst_defect_type,
        "worst_defect_score": worst_defect_score,
    }


# =========================
# 12. Single-image tests (SV + RS)
# =========================

# Example SV image
test_path = "/content/drive/MyDrive/25Fall/25Fall class/generative ai/Dataset/sample/SVI.jpg"

res = process_image_with_gemini_agent(
    test_path,
    hazard_type="wildfire",
    delta_Q=0.0,
)

print("source_type:", res["source_type"])
print("detected_problems:", res["detected_problems"])
print("rule_problems:", res["rule_problems"])
print("final_problems:", res.get("final_problems", {}))
print("tools_tried:", res.get("tools_tried", []))
print("recommended_tools (accepted):", res["recommended_tools"])
print("accepted:", res["accepted"])
print("Q_before (Gemini pipeline):", res["Q_before"])
print("Q_after  (Gemini pipeline):", res["Q_after"])

base_res = compute_Q_base_for_image(
    test_path,
    source_type_override=res["source_type"],
    save_base_image=False,
    suffix="_base",
)

print("\n=== Baseline (Q_base) ===")
print("Q_before_base:", base_res["Q_before_base"])
print("Q_base       :", base_res["Q_base"])
print("base_tools   :", base_res["base_tools"])
print("worst_defect_type :", base_res["worst_defect_type"])
print("worst_defect_score:", base_res["worst_defect_score"])

if res["accepted"] == 1 and len(res["recommended_tools"]) > 0:
    img_rgb = load_image_rgb(test_path)
    restored_rgb = apply_tool_chain(img_rgb, res["recommended_tools"])

    folder = os.path.dirname(test_path)
    base_name, ext = os.path.splitext(os.path.basename(test_path))
    out_path = os.path.join(folder, base_name + "_restored" + ext)

    out_bgr = cv2.cvtColor(
        (restored_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(out_path, out_bgr)

    print("\nRestored image (Gemini) saved to:", out_path)
else:
    print("\nRestoration not accepted; no restored image saved.")


# Example RS image
test_path = (
    "/content/drive/MyDrive/25Fall/25Fall class/generative ai/Dataset/sample/Satellite.png"
)

res = process_image_with_gemini_agent(
    test_path,
    hazard_type="wildfire",
    delta_Q=0.0,
)

print("\n=== RS example ===")
print("source_type:", res["source_type"])
print("detected_problems:", res["detected_problems"])
print("rule_problems:", res["rule_problems"])
print("final_problems:", res.get("final_problems", {}))
print("tools_tried:", res.get("tools_tried", []))
print("recommended_tools (accepted):", res["recommended_tools"])
print("accepted:", res["accepted"])
print("Q_before (Gemini pipeline):", res["Q_before"])
print("Q_after  (Gemini pipeline):", res["Q_after"])

base_res = compute_Q_base_for_image(
    test_path,
    source_type_override=res["source_type"],
    save_base_image=True,
    suffix="_base",
)

print("\n=== Baseline (Q_base, RS) ===")
print("Q_before_base:", base_res["Q_before_base"])
print("Q_base       :", base_res["Q_base"])
print("base_tools   :", base_res["base_tools"])
print("worst_defect_type :", base_res["worst_defect_type"])
print("worst_defect_score:", base_res["worst_defect_score"])
print("base_image_path   :", base_res["base_image_path"])

if res["accepted"] == 1 and len(res["recommended_tools"]) > 0:
    img_rgb = load_image_rgb(test_path)
    restored_rgb = apply_tool_chain(img_rgb, res["recommended_tools"])

    folder = os.path.dirname(test_path)
    base_name, ext = os.path.splitext(os.path.basename(test_path))
    out_path = os.path.join(folder, base_name + "_restored" + ext)

    out_bgr = cv2.cvtColor(
        (restored_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(out_path, out_bgr)

    print("\nRestored image (Gemini) saved to:", out_path)
else:
    print("\nRestoration not accepted; no restored image saved.")


# =========================
# 13. Batch processing with Gemini agent + baseline
# =========================

def batch_process_with_gemini_agent(
    root_folder,
    csv_out_path,
    default_hazard="unknown",
    delta_Q=0.01,
    save_restored=False,
    restored_root=None,
    save_base_image=True,
):
    """
    Recursively process all images under root_folder.

    For each image:
      1) Compute Q_base via the baseline pipeline.
      2) Run the Gemini-based agent.
      3) Optionally save the restored image.
      4) Flatten all results and write one row into a CSV.

    The output CSV contains, among others:
      - Q_before, Q_after (Gemini agent)
      - Q_before_base, Q_base (baseline)
      - restored_path (Gemini)
      - base_image_path (baseline)
    """
    if save_restored and restored_root is None:
        restored_root = root_folder.rstrip("/\\") + "_restored"

    rows = []

    print("Scanning:", root_folder)
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)

            ext = os.path.splitext(fname)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                continue

            if fname.lower().startswith("label") and ext in [".tif", ".tiff"]:
                print("[Skip label]:", fpath)
                continue

            try:
                hazard_type = default_hazard

                # 1) Baseline Q_base
                base_res = compute_Q_base_for_image(
                    fpath,
                    source_type_override=None,
                    save_base_image=save_base_image,
                    suffix="_base",
                )

                # 2) Gemini agent
                res = process_image_with_gemini_agent(
                    fpath,
                    hazard_type=hazard_type,
                    delta_Q=delta_Q,
                )

                # 3) Optional: save Gemini-restored image
                restored_path = ""
                if (
                    save_restored
                    and res["accepted"] == 1
                    and len(res["recommended_tools"]) > 0
                ):
                    img_rgb = load_image_rgb(fpath)
                    restored_rgb = apply_tool_chain(
                        img_rgb, res["recommended_tools"]
                    )

                    rel_path = os.path.relpath(fpath, root_folder)
                    rel_dir = os.path.dirname(rel_path)
                    base_name, ext2 = os.path.splitext(
                        os.path.basename(rel_path)
                    )

                    out_dir = os.path.join(restored_root, rel_dir)
                    os.makedirs(out_dir, exist_ok=True)

                    out_path = os.path.join(
                        out_dir, base_name + "_restored" + ext2
                    )
                    out_bgr = cv2.cvtColor(
                        (restored_rgb * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    )
                    cv2.imwrite(out_path, out_bgr)

                    restored_path = out_path

                # 4) Flatten row
                row = {
                    "image": res["image"],
                    "path": res["path"],
                    "folder": os.path.dirname(res["path"]),
                    "source_type": res["source_type"],
                    "hazard_type": res["hazard_type"],
                    "detected_problems_gemini": ";".join(
                        res["detected_problems"]
                    )
                    if res["detected_problems"]
                    else "",
                    "tools_tried": ";".join(res.get("tools_tried", [])),
                    "recommended_tools": ";".join(
                        res.get("recommended_tools", [])
                    ),
                    "agent_decision": res["agent_decision"],
                    "accepted": res["accepted"],
                    "Q_before": res["Q_before"],
                    "Q_after": res["Q_after"],
                    "delta_Q": res["delta_Q"],
                    "restored_path": restored_path,
                    # Baseline (Q_base)
                    "Q_before_base": base_res["Q_before_base"],
                    "Q_base": base_res["Q_base"],
                    "base_tools": ";".join(base_res.get("base_tools", [])),
                    "base_image_path": base_res.get("base_image_path", None),
                    "base_worst_defect_type": base_res.get(
                        "worst_defect_type", None
                    ),
                    "base_worst_defect_score": base_res.get(
                        "worst_defect_score", None
                    ),
                }

                for k, v in base_res.get("defect_scores", {}).items():
                    row[f"base_defect_{k}"] = v

                for k, v in res.get("rule_problems", {}).items():
                    row[f"rule_{k}"] = int(bool(v))

                if "final_problems" in res:
                    for k, v in res["final_problems"].items():
                        row[f"final_{k}"] = int(bool(v))

                fb = res["features_before"]
                fa = res["features_after"]
                for k, v in fb.items():
                    row[f"before_{k}"] = v
                for k, v in fa.items():
                    row[f"after_{k}"] = v

                rows.append(row)

            except Exception as e:
                print(f"[Error] {fpath}: {e}")
                continue

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_out_path), exist_ok=True)
    df.to_csv(csv_out_path, index=False, encoding="utf-8-sig")

    print("Saved CSV:", csv_out_path)
    print("Total processed images:", len(df))

    return df


root_folder = "/content/drive/MyDrive/25Fall/25Fall class/generative ai/Dataset/sample/SVI_PalisadesFireImages"
csv_out_path = "/content/drive/MyDrive/25Fall/25Fall class/generative ai/Dataset/sample/SVI_PalisadesFireImages/agent_results_gemini.csv"

df_results = batch_process_with_gemini_agent(
    root_folder=root_folder,
    csv_out_path=csv_out_path,
    default_hazard="wildfire",
    delta_Q=0.00,
    save_restored=True,
    restored_root=None,
)

df_results.head()

df = df_results.copy()
df["deltaQ_gemini"] = df["Q_after"] - df["Q_before"]
df["deltaQ_base"] = df["Q_base"] - df["Q_before"]
df["deltaQ_vs_base"] = df["Q_after"] - df["Q_base"]

summary = pd.DataFrame(
    {
        "mean_Q_before": [df["Q_before"].mean()],
        "mean_Q_after": [df["Q_after"].mean()],
        "mean_Q_base": [df["Q_base"].mean()],
        "mean_deltaQ_gemini": [df["deltaQ_gemini"].mean()],
        "mean_deltaQ_base": [df["deltaQ_base"].mean()],
        "mean_deltaQ_vs_base": [df["deltaQ_vs_base"].mean()],
        "improved_gemini": [(df["deltaQ_gemini"] > 0).mean()],
        "improved_base": [(df["deltaQ_base"] > 0).mean()],
        "better_than_base": [(df["deltaQ_vs_base"] > 0).mean()],
    }
)

summary.T
