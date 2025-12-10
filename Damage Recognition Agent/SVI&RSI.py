import os
import glob
import json
import time
import pandas as pd
from PIL import Image
from google import genai
from google.genai import types

# --- 1. Configuration ---

# Ensure API Key is set
os.environ["GEMINI_API_KEY"] = "API"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Dataset root directory path (Code will automatically scan images in this directory)
DATASET_ROOT_PATH = ".../Dataset/SVI & RSI_CVIAN_selected_50"

# Output Excel filename (Recommended to save to Drive)
OUTPUT_EXCEL_FILE = ".../Dataset/gemini-2.5-pro_SVI & RSI_CVIAN.xlsx"

# --- 2. Core Assessment Function ---

def assess_multimodal_pair(image_id, rsi_path, svi_path):
    """
    Analyzes a pair of images (Satellite + Street View) using Gemini.
    Returns a dictionary containing classification, object detection, and confidence scores.
    """

    # Define strict grading criteria
    grading_criteria = """
    - **Mild**: Images characterized by clean scenes with no major property damage or only minor disruptions (such as small areas of fallen trees) and are considered to have caused negligible economic loss.
    - **Moderate**: Images exhibit more impacts compared to "Mild", typically featuring more pronounced fallen trees, some visible building debris, and visible pools of water around the base of trees. Indicates noticeable economic damage.
    - **Severe**: The most chaotic cases, distinguished by extensive or widespread tree falls, fully flooded roads, and significant building debris. Represents major destruction.
    """

    # Construct Prompt
    prompt = f"""
    You are an expert Disaster Assessment AI Agent.
    You are provided with two images of the SAME location after a hurricane:
    1. A Remote Sensing Image (Satellite/Aerial view).
    2. A Street View Image (Ground-level view).

    **Task 1: Damage Grading**
    Analyze both images to determine the overall damage severity based on these strict criteria:
    {grading_criteria}

    **Task 2: Object Verification (Street Level Focus)**
    Detect the presence of specific objects (0 = No/Unsure, 1 = Yes/Clearly Visible).

    **Output Requirement:**
    Return a strictly valid JSON object with the following structure:
    {{
        "Predicted_Severity": "Mild" or "Moderate" or "Severe",
        "Confidence_Score": <float between 0.0 and 1.0>,
        "Objects": {{
            "debris_pile": 0 or 1,
            "fallen_tree": 0 or 1,
            "flooded_road": 0 or 1,
            "damaged_building": 0 or 1,
            "downed_lines": 0 or 1
        }},
        "Reasoning": "<Short explanation citing evidence from both RS and SVI>"
    }}
    """

    try:
        # Load Images
        img_rs = Image.open(rsi_path)
        img_svi = Image.open(svi_path)

        # Resize to optimize transmission
        img_rs.thumbnail([1024, 1024])
        img_svi.thumbnail([1024, 1024])

        # Call API
        response = client.models.generate_content(
            model="gemini-2.5-pro", # Recommended to use flash-exp or pro for speed and stability
            contents=[prompt, img_rs, img_svi], # Input both Satellite and Street View images
            config=types.GenerateContentConfig(
                temperature=0.0, # Zero temperature to ensure deterministic results
                response_mime_type="application/json",
                # --- Critical Fix: Disable safety filters to prevent rejection of debris images ---
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                ]
            )
        )

        # --- Critical Fix: Check if response is empty ---
        if not response.text:
            print(f"⚠️ Warning: Pair {image_id} returned empty response (Blocked by safety filters).")
            return {
                "Predicted_Severity": "Blocked",
                "Confidence_Score": 0.0,
                "Objects": {"debris_pile": 0, "fallen_tree": 0, "flooded_road": 0, "damaged_building": 0, "downed_lines": 0},
                "Reasoning": "Model safety block triggered."
            }

        return json.loads(response.text)

    except Exception as e:
        print(f"Error processing pair {image_id}: {e}")
        # Return default error structure to prevent program crash
        return {
            "Predicted_Severity": "Error",
            "Confidence_Score": 0.0,
            "Objects": {"debris_pile": 0, "fallen_tree": 0, "flooded_road": 0, "damaged_building": 0, "downed_lines": 0},
            "Reasoning": str(e)
        }

# --- 3. Batch Processing Logic ---

def process_dataset(root_path):
    results = []

    # Recursively search for all png images
    print(f"Scanning directory: {root_path} ...")
    all_files = glob.glob(os.path.join(root_path, "**/*.png"), recursive=True)

    # Extract unique IDs
    unique_ids = set()
    file_map = {} # Map for quick path lookup

    for f_path in all_files:
        f_name = os.path.basename(f_path)
        if "_Satellite" in f_name:
            uid = f_name.replace("_Satellite.png", "")
            unique_ids.add(uid)
            file_map[f"{uid}_RS"] = f_path
        elif "_SVI" in f_name:
            uid = f_name.replace("_SVI.png", "")
            file_map[f"{uid}_SVI"] = f_path

    print(f"Found {len(unique_ids)} unique location pairs.")

    # Iterate through each ID for assessment
    for i, uid in enumerate(unique_ids):
        rs_path = file_map.get(f"{uid}_RS")
        svi_path = file_map.get(f"{uid}_SVI")

        # Process only when both images exist
        if rs_path and svi_path:
            print(f"[{i+1}/{len(unique_ids)}] Processing ID: {uid} ...")

            # Call core function
            analysis = assess_multimodal_pair(uid, rs_path, svi_path)

            # Flatten data structure for Excel storage
            record = {
                "Location_ID": uid,
                "Input_RSI": os.path.basename(rs_path),
                "Input_SVI": os.path.basename(svi_path),
                "Voted_Candidate_Severity": analysis.get("Predicted_Severity", "Unknown"),
                "Confidence": analysis.get("Confidence_Score", 0),
                # Expand Object dictionary
                "Obj_Debris": analysis["Objects"].get("debris_pile", 0),
                "Obj_FallenTree": analysis["Objects"].get("fallen_tree", 0),
                "Obj_Flooded": analysis["Objects"].get("flooded_road", 0),
                "Obj_DamagedBldg": analysis["Objects"].get("damaged_building", 0),
                "Obj_DownedLines": analysis["Objects"].get("downed_lines", 0),
                "Reasoning": analysis.get("Reasoning", "")
            }
            results.append(record)

            # Avoid API rate limits
            time.sleep(1)
        else:
            print(f"Skipping {uid}: Missing pair (RS or SVI not found).")

    return results

# --- 4. Execution and Saving ---

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT_PATH):
        # Start processing
        data_records = process_dataset(DATASET_ROOT_PATH)

        if data_records:
            # Create DataFrame
            df = pd.DataFrame(data_records)

            # Save as Excel
            df.to_excel(OUTPUT_EXCEL_FILE, index=False)

            print("\n" + "="*50)
            print(f"Success! Processed {len(df)} pairs.")
            print(f"Results saved to: {OUTPUT_EXCEL_FILE}")
            print("="*50)

            # Display first few rows in Colab
            try:
                from IPython.display import display
                display(df.head())
            except:
                pass
        else:
            print("No valid image pairs found.")
    else:
        print(f"❌ Error: Root path not found {DATASET_ROOT_PATH}")
