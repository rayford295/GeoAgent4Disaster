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
os.environ["GEMINI_API_KEY"] = "Your API"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Dataset Path
DATASET_ROOT_PATH = "/content/drive/MyDrive/Manuscript_Ph.D./3rd_DisasterMind AI agent (organization）/Dataset/pre and post SVI(50-50-50)"

# Output Excel Filename
OUTPUT_EXCEL_FILE = "/content/drive/MyDrive/Manuscript_Ph.D./3rd_DisasterMind AI agent (organization）/Dataset/gemini-3-pro_PrePost_SVI_Assessment_Results_Fixed2023_2024.xlsx"

# --- 2. Core Assessment Function ---

def assess_pre_post_pair(folder_name, pre_img_path, post_img_path):
    """
    Function to analyze a pair of Pre-disaster (2023) and Post-disaster (2024) images.
    """

    grading_criteria = """
    - **Mild**: Minimal visible change. Some fallen branches or minor debris, but infrastructure and buildings remain largely intact compared to the 2023 baseline.
    - **Moderate**: Noticeable damage. Visible structural damage to roofs/facades, significant debris piles, or trees blocking paths that were previously clear in the 2023 image.
    - **Severe**: Drastic transformation. Complete destruction of buildings, massive flooding, or roads becoming completely impassable due to debris/trees compared to the 2023 state.
    """

    # --- Modified Prompt: Clearly specify image identity ---
    prompt = f"""
    You are an expert Disaster Assessment AI.
    You are provided with two Street View Images of the SAME location.

    **Input Structure:**
    - **Image 1**: Pre-disaster baseline (Taken in 2023).
    - **Image 2**: Post-disaster situation (Taken in 2024).

    **Your Tasks:**
    1. **Comparative Grading**: Compare the Post-disaster image (Image 2) against the Pre-disaster baseline (Image 1). Assess the severity of the *change* and damage based on:
    {grading_criteria}

    2. **Object Detection (In the Post-Disaster Image Only)**: Detect specific disaster objects (0=No, 1=Yes).

    **Output Requirement:**
    Return a strictly valid JSON object:
    {{
        "Predicted_Severity": "Mild" or "Moderate" or "Severe",
        "Confidence_Score": <float 0.0-1.0>,
        "Objects_Detected": {{
            "debris_pile": 0 or 1,
            "fallen_tree": 0 or 1,
            "flooded_road": 0 or 1,
            "damaged_building": 0 or 1,
            "downed_lines": 0 or 1
        }},
        "Reasoning": "<Explain the key changes observed from 2023 to 2024>"
    }}
    """

    try:
        # Load Images
        img_pre = Image.open(pre_img_path)
        img_post = Image.open(post_img_path)

        # Uniform Resizing
        img_pre.thumbnail([1024, 1024])
        img_post.thumbnail([1024, 1024])

        # Call API (Fixed order: Pre first, then Post)
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[prompt, img_pre, img_post],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                ]
            )
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"Error processing {folder_name}: {e}")
        return None

# --- 3. Batch Processing Logic (Updated: Identify 2023/2024 by filename) ---

def process_pre_post_dataset(root_path):
    results = []

    print(f"Scanning directory: {root_path} ...")

    for dirpath, dirnames, filenames in os.walk(root_path):
        images = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(images) == 2:
            # --- Critical Modification: Identify Pre and Post by filename ---
            pre_img_name = None
            post_img_name = None

            for img in images:
                if "2023" in img:
                    pre_img_name = img
                elif "2024" in img:
                    post_img_name = img

            # Only process if both years are found
            if pre_img_name and post_img_name:
                folder_name = os.path.basename(dirpath)
                parent_category = os.path.basename(os.path.dirname(dirpath)) # mild/moderate/severe

                pre_img_path = os.path.join(dirpath, pre_img_name)
                post_img_path = os.path.join(dirpath, post_img_name)

                print(f"Processing Pair: {parent_category}/{folder_name} (Pre: 2023, Post: 2024) ...")

                # Call Assessment
                analysis = assess_pre_post_pair(folder_name, pre_img_path, post_img_path)

                if analysis:
                    record = {
                        "Pair_ID": folder_name,
                        "Ground_Truth_Label": parent_category,
                        "Pre_Image_File": pre_img_name,
                        "Post_Image_File": post_img_name,
                        "Predicted_Severity": analysis.get("Predicted_Severity", "Unknown"),
                        "Confidence": analysis.get("Confidence_Score", 0),
                        # Objects
                        "Obj_Debris": analysis["Objects_Detected"].get("debris_pile", 0),
                        "Obj_FallenTree": analysis["Objects_Detected"].get("fallen_tree", 0),
                        "Obj_Flooded": analysis["Objects_Detected"].get("flooded_road", 0),
                        "Obj_DamagedBldg": analysis["Objects_Detected"].get("damaged_building", 0),
                        "Obj_DownedLines": analysis["Objects_Detected"].get("downed_lines", 0),
                        "Reasoning": analysis.get("Reasoning", "")
                    }
                    results.append(record)
                    time.sleep(1)
                else:
                    print(f"Failed analysis for {folder_name}")
            else:
                # If corresponding year tags are not found
                print(f"Skipping folder {os.path.basename(dirpath)}: Filenames must contain '2023' and '2024'. Found: {images}")

    return results

# --- 4. Execution and Saving ---

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT_PATH):
        records = process_pre_post_dataset(DATASET_ROOT_PATH)

        if records:
            df = pd.DataFrame(records)
            df.to_excel(OUTPUT_EXCEL_FILE, index=False)
            print("\n" + "="*50)
            print(f"✅ Completed! Processed {len(df)} pairs.")
            print(f"📄 Results saved to: {OUTPUT_EXCEL_FILE}")

            try:
                from IPython.display import display
                display(df.head())
            except: pass
        else:
            print("⚠️ No valid pairs found. Please check if filenames contain '2023' and '2024'.")
    else:
        print(f"❌ Path does not exist: {DATASET_ROOT_PATH}")
