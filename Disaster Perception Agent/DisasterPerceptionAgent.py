import os
import json
import shutil
from typing import List, Dict, Any, Union, Tuple, Optional
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from openai import OpenAI

# =========================
#   0. OpenAI Client
# =========================

OPENAI_API_KEY = "KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

# ========== Global utilities ==========
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def ensure_pil_image(x: Union[str, Image.Image]) -> Image.Image:
    """Ensure the input is a PIL.Image.Image; if it is a path string, open it as RGB."""
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, str):
        img = Image.open(x).convert("RGB")
        return img
    raise TypeError(f"Unsupported image type: {type(x)}")


def pil_to_data_url(img: Image.Image, format: str = "JPEG") -> str:
    """Convert a PIL Image into a base64 data URL for OpenAI Vision."""
    buf = BytesIO()
    img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"


def openai_json_call(
    model: str,
    user_content: List[Dict[str, Any]],
    system_instruction: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Call the OpenAI Responses API and force the output to be a JSON object.
    user_content: List[{"type": "input_text"/"input_image", ...}]
    """
    resp = client.responses.create(
        model=model,
        instructions=system_instruction,
        input=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
        temperature=temperature,
    )

    try:
        text = resp.output[0].content[0].text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to parse text from Responses output: {e}")

    try:
        data = json.loads(text)
    except Exception as e:
        print("Failed to parse JSON, raw output is:")
        print(text)
        raise e
    return data


class ModePerceiver:
    """
    Multi-modal zero-shot mode and hazard type recognition.

    Modes:
        - post_sv_single
        - post_paired_sv_rs
        - bitemporal_sv
        - bitemporal_rs

    Hazard types:
        - hurricane
        - flood
        - wildfire
        - earthquake
        - tornado
        - landslide
        - other
    """

    MODES = [
        "post_sv_single",
        "post_paired_sv_rs",
        "bitemporal_sv",
        "bitemporal_rs",
    ]

    HAZARDS = [
        "hurricane",
        "flood",
        "wildfire",
        "earthquake",
        "tornado",
        "landslide",
        "other",
    ]

    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name

    def infer(
        self, images: Union[Image.Image, List[Image.Image]]
    ) -> Dict[str, Any]:
        if not isinstance(images, list):
            images = [images]

        pil_images = [ensure_pil_image(im) for im in images]

        # Build user_content (mixed text + image)
        content: List[Dict[str, Any]] = []
        for idx, im in enumerate(pil_images):
            content.append(
                {
                    "type": "input_image",
                    "image_url": pil_to_data_url(im),
                }
            )
            content.append(
                {
                    "type": "input_text",
                    "text": f"Image index: {idx}. This is one of the input disaster-related images.",
                }
            )

        mode_description = "\n".join(
            [
                "1. post_sv_single    - a single post-disaster street-view image.",
                "2. post_paired_sv_rs - a pair of post-disaster images: one street-view and one satellite image.",
                "3. bitemporal_sv     - a pair of street-view images of the same place before and after a disaster.",
                "4. bitemporal_rs     - a pair of satellite images of the same area before and after a disaster.",
            ]
        )

        hazard_description = "\n".join(
            [
                "Supported hazard types:",
                "- hurricane: tropical cyclone, storm surge, strong wind and rain;",
                "- flood: river flood, urban flood, inundation, standing water;",
                "- wildfire: forest fire, bushfire, open fire with smoke plumes;",
                "- earthquake: collapsed buildings, ground rupture, rubble without clear fire or flood;",
                "- tornado: narrow corridor wind damage, twisted debris, funnel cloud;",
                "- landslide: slope failure, soil or rock movement, buried roads or houses;",
                "- other: if none of the above clearly fits.",
            ]
        )

        system_instruction = (
            "You are a disaster image analyst. "
            "Given 1-2 images of the same scene, you must:\n"
            "1) Choose exactly ONE of the four image modes;\n"
            "2) Identify the most likely disaster type;\n"
            "3) For each image, judge whether it visually needs restoration "
            "(for example, low resolution, motion blur, compression artifacts, severe noise, over/under exposure);\n"
            "4) Return a compact JSON object ONLY, with no extra text.\n\n"
            f"{mode_description}\n\n{hazard_description}\n\n"
            "Return JSON with keys:\n"
            "{\n"
            '  "pred_mode": string,   # one of the four modes\n'
            '  "mode_confidence": float,  # 0-1\n'
            '  "mode_probs": {mode_name: float},\n'
            '  "hazard_type": string,      # one of the hazard labels\n'
            '  "hazard_confidence": float, # 0-1\n'
            '  "needs_restoration": [bool, ...],  # length = number of images\n'
            '  "restoration_reasons": [string, ...]  # same length as images\n'
            "}\n"
            "Do NOT include explanations outside the JSON. JSON must be valid and parseable."
        )

        data = openai_json_call(
            model=self.model_name,
            user_content=content,
            system_instruction=system_instruction,
            temperature=0.2,
        )

        # Simple fault tolerance / defaults
        data.setdefault("pred_mode", "post_sv_single")
        data.setdefault("mode_confidence", 0.0)
        data.setdefault("mode_probs", {})
        data.setdefault("hazard_type", "other")
        data.setdefault("hazard_confidence", 0.0)
        n_img = len(pil_images)
        needs = data.get("needs_restoration") or [False] * n_img
        reasons = data.get("restoration_reasons") or [""] * n_img
        if len(needs) != n_img:
            needs = (needs + [False] * n_img)[:n_img]
        if len(reasons) != n_img:
            reasons = (reasons + [""] * n_img)[:n_img]
        data["needs_restoration"] = needs
        data["restoration_reasons"] = reasons

        return data


class DisasterReasoner:
    """
    Use chain-of-thought style reasoning to generate detailed descriptive
    disaster reasoning text based on the predicted mode and hazard type.
    """

    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name

    def generate_reasoning(
        self,
        images: Union[Image.Image, List[Image.Image]],
        mode_result: Dict[str, Any],
    ) -> str:
        if not isinstance(images, list):
            images = [images]
        pil_images = [ensure_pil_image(im) for im in images]

        content: List[Dict[str, Any]] = []
        for idx, im in enumerate(pil_images):
            content.append(
                {
                    "type": "input_image",
                    "image_url": pil_to_data_url(im),
                }
            )
            content.append(
                {
                    "type": "input_text",
                    "text": f"Image index {idx}. Use this for detailed disaster reasoning.",
                }
            )

        meta_text = json.dumps(
            {"mode_result": mode_result},
            ensure_ascii=False,
            indent=2,
        )
        content.append(
            {
                "type": "input_text",
                "text": "Structured metadata about the images:\n" + meta_text,
            }
        )

        system_instruction = (
            "You are an expert disaster analyst. "
            "Your task is to produce a very detailed, step-by-step reasoning text "
            "that describes all observable disaster-related damage in the input image(s).\n\n"
            "Write in natural, continuous prose (multi-paragraph text). "
            "Do NOT organize the output by object categories or headings such as 'Roads' "
            "or 'Buildings'. Avoid Markdown lists and bullet points.\n\n"
            "Guidelines:\n"
            "1. First, briefly restate the predicted image mode and disaster type in 1-2 sentences.\n"
            "2. Then, describe the overall scene layout (for example, roads, buildings, vegetation, water, sky) "
            "to give a global context.\n"
            "3. After that, carefully describe each damaged element that you can see in the scene "
            "(for example damaged houses, broken roofs, flooded streets, debris piles, fallen trees, "
            "collapsed infrastructure, standing water, smoke, etc.), but weave them into a continuous narrative "
            "instead of separating them into sections.\n"
            "   For each damaged element you mention, explain:\n"
            "   - what kind of object it is and how it appears visually (color, texture, shape, size),\n"
            "   - what kind of disaster damage is visible on it (for example, collapsed, cracked, burnt, submerged),\n"
            "   - where it is approximately located in the image (for example, upper-right area, near the center road, "
            "     along the shoreline, in the foreground or background).\n"
            "4. Be explicit and granular so that an engineer could later extract a rich vocabulary of visual cues "
            "and damaged object types from your text.\n"
            "5. At the end, include a short concluding paragraph that summarizes the main types of damaged objects "
            "and the overall severity of damage observed in the scene.\n"
            "6. Do NOT output JSON or lists. Output only a readable multi-paragraph text in English.\n"
        )

        resp = client.responses.create(
            model=self.model_name,
            instructions=system_instruction,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=0.4,
        )

        try:
            text = resp.output[0].content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to parse text from Responses output: {e}")
        return text


class TaskPlanner:
    """
    Task planner that consumes:
      - mode_result (from ModePerceiver)
      - reasoning_text (from DisasterReasoner)
    and constructs a concise tool-use plan specifying which downstream
    agents to invoke (e.g., ImageRestorationAgent, DamageRecognitionAgent).
    """

    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name

    def plan(
        self,
        mode_result: Dict[str, Any],
        reasoning_text: str,
        image_paths: List[str],
        view_types: List[str],
        needs_restoration: List[bool],
        copied_paths: List[str],
    ) -> Dict[str, Any]:
        # 为避免 prompt 过长，这里截断 reasoning_text
        truncated_reasoning = reasoning_text[:2000]

        planner_input = {
            "mode_result": mode_result,
            "reasoning_text_truncated": truncated_reasoning,
            "image_paths": image_paths,
            "view_types": view_types,
            "needs_restoration": needs_restoration,
            "restoration_copied_paths": copied_paths,
        }

        content: List[Dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "You are a task planner in a multi-agent GeoAI system for natural disaster assessment.\n\n"
                    "Upstream modules have already produced:\n"
                    "- mode_result: prediction of image mode, hazard type, and which images need restoration.\n"
                    "- reasoning_text: a detailed narrative description of observed damage.\n\n"
                    "Downstream agents available:\n"
                    "1. ImageRestorationAgent  – improves image quality for inputs that visually need restoration.\n"
                    "2. DamageRecognitionAgent – performs structured damage recognition / segmentation / classification.\n"
                    "3. DisasterReasoningAgent – already run in this pipeline; you usually do NOT need to re-run it.\n\n"
                    "Your job is to construct a concise tool-use plan that decides:\n"
                    "- whether ImageRestorationAgent should be invoked,\n"
                    "- for which images it should be invoked (by index, view_type, and path),\n"
                    "- whether DamageRecognitionAgent should be invoked and with what main inputs.\n\n"
                    "Here is the structured input from upstream modules:\n"
                    + json.dumps(planner_input, ensure_ascii=False, indent=2)
                ),
            }
        ]

        system_instruction = (
            "You are a careful planner. Based on the provided JSON, decide which downstream agents to run.\n"
            "Typically:\n"
            "- If any image in needs_restoration is true, you should include ImageRestorationAgent.\n"
            "- If hazard_type is not 'other' or the reasoning_text describes visible damage, you should include DamageRecognitionAgent.\n"
            "- DisasterReasoningAgent has already been executed upstream; only include it if you strongly believe a second pass is necessary.\n\n"
            "Return a SINGLE JSON object with the following structure:\n"
            "{\n"
            '  \"downstream_agents\": [\"ImageRestorationAgent\", \"DamageRecognitionAgent\", ...],\n'
            '  \"should_invoke_restoration\": bool,\n'
            '  \"restoration_targets\": [\n'
            "    {\n"
            '      \"image_index\": int,\n'
            '      \"view_type\": \"sv\" | \"rs\",\n'
            '      \"original_path\": string,\n'
            '      \"copied_path\": string | null,\n'
            '      \"reason\": string\n'
            "    }, ...\n"
            "  ],\n"
            '  \"should_invoke_damage_recognition\": bool,\n'
            '  \"damage_recognition_inputs\": {\n'
            '    \"image_paths\": [string, ...],\n'
            '    \"priority_notes\": string\n'
            "  },\n"
            '  \"planner_notes\": string  # short natural language explanation of the plan\n'
            "}\n\n"
            "Only output valid JSON with no extra commentary."
        )

        data = openai_json_call(
            model=self.model_name,
            user_content=content,
            system_instruction=system_instruction,
            temperature=0.2,
        )

        # 简单兜底
        data.setdefault("downstream_agents", [])
        data.setdefault("should_invoke_restoration", any(needs_restoration))
        data.setdefault("restoration_targets", [])
        data.setdefault("should_invoke_damage_recognition", True)
        data.setdefault(
            "damage_recognition_inputs",
            {
                "image_paths": image_paths,
                "priority_notes": "Fallback default plan.",
            },
        )
        data.setdefault("planner_notes", "Fallback default plan generated by TaskPlanner.")

        return data


def infer_view_types(pred_mode: str, num_images: int) -> List[str]:
    """
    Infer whether each image is street-view (sv) or remote-sensing (rs)
    based on the predicted mode.
    """
    if pred_mode == "post_sv_single":
        return ["sv"]
    if pred_mode == "post_paired_sv_rs":
        return ["sv", "rs"][:num_images]
    if pred_mode == "bitemporal_sv":
        return ["sv"] * num_images
    if pred_mode == "bitemporal_rs":
        return ["rs"] * num_images
    return ["sv"] * num_images  # fallback


def copy_images_for_restoration(
    image_paths: List[str],
    view_types: List[str],
    needs_restoration: List[bool],
    out_root: str = "to_restore",
) -> List[str]:
    """
    Copy images that need restoration into:
      - to_restore/sv
      - to_restore/rs
    """
    sv_dir = Path(out_root) / "sv"
    rs_dir = Path(out_root) / "rs"
    sv_dir.mkdir(parents=True, exist_ok=True)
    rs_dir.mkdir(parents=True, exist_ok=True)

    copied_paths: List[str] = []
    for path, vt, need in zip(image_paths, view_types, needs_restoration):
        if not need:
            continue
        dst_dir = rs_dir if vt == "rs" else sv_dir
        base = os.path.basename(path)
        dst_path = dst_dir / base
        shutil.copy2(path, dst_path)
        copied_paths.append(str(dst_path))

    return copied_paths


class DisasterPerceptionAgent:
    def __init__(
        self,
        mode_perceiver: ModePerceiver,
        reasoner: DisasterReasoner,
        task_planner: TaskPlanner,
    ):
        self.mode_perceiver = mode_perceiver
        self.reasoner = reasoner
        self.task_planner = task_planner

    def run(
        self,
        images: Union[Image.Image, List[Image.Image]],
        image_paths: List[str],
        run_name: str = "case",
    ) -> Dict[str, Any]:
        """
        Main entry point:
          1. Recognize mode and hazard type and whether restoration is needed;
          2. Generate detailed reasoning text based on mode and hazard;
          3. Copy images that need restoration into to_restore/sv and to_restore/rs;
          4. Call TaskPlanner to construct a concise tool-use plan.
        """
        if not isinstance(images, list):
            images = [images]
        pil_images = [ensure_pil_image(im) for im in images]

        # 1) Mode, hazard, and restoration requirement
        mode_result = self.mode_perceiver.infer(pil_images)

        # 2) Reasoning text (CoT)
        reasoning_text = self.reasoner.generate_reasoning(
            pil_images,
            mode_result=mode_result,
        )

        # 3) Restoration info
        n_img = len(pil_images)
        view_types = infer_view_types(mode_result.get("pred_mode", ""), n_img)
        needs_restoration = mode_result.get("needs_restoration", [False] * n_img)
        if len(needs_restoration) != n_img:
            needs_restoration = (needs_restoration + [False] * n_img)[:n_img]

        copied_paths = copy_images_for_restoration(
            image_paths=image_paths,
            view_types=view_types,
            needs_restoration=needs_restoration,
            out_root="to_restore",
        )

        restoration_info = {
            "view_types": view_types,
            "needs_restoration": needs_restoration,
            "copied_paths": copied_paths,
        }

        # 4) Task planning (consumes outputs of previous two modules)
        tool_plan = self.task_planner.plan(
            mode_result=mode_result,
            reasoning_text=reasoning_text,
            image_paths=image_paths,
            view_types=view_types,
            needs_restoration=needs_restoration,
            copied_paths=copied_paths,
        )

        return {
            "mode_and_hazard": mode_result,
            "reasoning_text": reasoning_text,
            "restoration": restoration_info,
            "tool_plan": tool_plan,
        }


def collect_cases_from_dataset(
    dataset_root: str,
    max_images: int = 2,
) -> List[Dict[str, Any]]:
    """
    Collect all cases from a three-level dataset directory structure.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    cases: List[Dict[str, Any]] = []

    # First level: dataset types
    for type_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        # Second level: hazard types / damage levels, etc. (only for organization, not used directly)
        for class_dir in sorted([d for d in type_dir.iterdir() if d.is_dir()]):
            # Third level: leaf folders or direct images
            leaf_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
            img_files = [
                f
                for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTS
            ]

            if leaf_dirs:
                # There are third-level subfolders: each leaf_dir is a "paired" case (for example, SV+RS or pre+post)
                for leaf in sorted(leaf_dirs):
                    imgs = [
                        str(p)
                        for p in sorted(leaf.iterdir())
                        if p.is_file() and p.suffix.lower() in IMG_EXTS
                    ]
                    if not imgs:
                        continue
                    cases.append(
                        {
                            "case_id": f"{type_dir.name}/{class_dir.name}/{leaf.name}",
                            "image_paths": imgs[:max_images],  # paired: at most max_images images
                        }
                    )
            elif img_files:
                # No third-level subfolders, only images:
                # each image is treated as a separate "non-paired" case
                for img in sorted(img_files):
                    cases.append(
                        {
                            "case_id": f"{type_dir.name}/{class_dir.name}/{img.stem}",
                            "image_paths": [str(img)],  # non-paired: single-image case
                        }
                    )
            else:
                # Neither leaf folders nor images; skip
                continue

    return cases


if __name__ == "__main__":
    # 1. Dataset root directory
    DATASET_ROOT = r"E:/TAMU study/Projects/21 AI agent/new/dataset/dataset_test"

    MAX_IMAGES_PER_CASE = 2  # Maximum number of images per case (for example, SV+RS)

    cases = collect_cases_from_dataset(DATASET_ROOT, max_images=MAX_IMAGES_PER_CASE)
    print(f"Found {len(cases)} cases")

    # 2. Initialize submodules and agent
    mode_perceiver = ModePerceiver(model_name="gpt-4.1-mini")
    reasoner = DisasterReasoner(model_name="gpt-4.1-mini")
    task_planner = TaskPlanner(model_name="gpt-4.1-mini")

    agent = DisasterPerceptionAgent(
        mode_perceiver=mode_perceiver,
        reasoner=reasoner,
        task_planner=task_planner,
    )

    # Output directory
    results_dir = Path("outputs/results1209")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 3. Run each case and write JSON
    for case in cases:
        case_id = case["case_id"]
        image_paths = case["image_paths"]
        print(f"\n===== Processing case: {case_id} =====")
        for p in image_paths:
            print(" -", p)

        images = [ensure_pil_image(p) for p in image_paths]

        result = agent.run(
            images=images,
            image_paths=image_paths,
            run_name=case_id,
        )

        # Build a more complete structure (with case_id and image paths)
        output_obj = {
            "case_id": case_id,
            "image_paths": image_paths,
            "result": result,
        }

        json_path = results_dir / f"{case_id.replace('/', '__')}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_obj, f, ensure_ascii=False, indent=2)

        print(f"JSON saved to: {json_path}")
