import os
import glob
import json
import time
import pandas as pd
import io  # <--- Key import added
from PIL import Image
from google import genai
from google.genai import types

# --- 1. Configuration Section ---
# Ensure API Key is set
os.environ["GEMINI_API_KEY"] = "API"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Dataset Path
DATASET_ROOT_PATH = ".../3rd_DisasterMind AI agent (organization）/Dataset/SVI_PalisadesFireImages"

# Output Result Path
OUTPUT_EXCEL_FILE = ".../Dataset/gemini-3-pro_Wildfire_Building_Classification_Results_NoReasoning.xlsx"

# --- 2. Core Classification Function ---

def classify_building_damage(image_path):
    """
    Perform building damage level classification only (5-class task), no reasoning process
    """

    # Define 5-level criteria
    classification_criteria = """
    Classify the damage to the PRIMARY residential structure into exactly one of these 5 categories:

    1. **0_No_Damage**: Structure is intact. No fire damage.
    2. **1_Affected_1_9**: Very minor cosmetic damage (soot, blistered paint). Structure stable.
    3. **2_Minor_10_25**: Visible non-structural damage (broken windows, melted siding). Repairable.
    4. **3_Major_26_50**: Significant structural damage (partial roof collapse, interior burn signs). Uninhabitable.
    5. **4_Destroyed_50plus**: Total loss. Collapsed or only chimney/foundation remaining.
    """

    # Minimalist Prompt: Selection only
    prompt = f"""
    You are a Wildfire Damage Classifier.

    Task: Identify the damage category for the main building in the image.

    Criteria:
    {classification_criteria}

    Instructions:
    - Ignore burnt trees if the building is fine.
    - Output strictly valid JSON.

    Output Format:
    {{
        "Predicted_Class": "0_No_Damage" or "1_Affected_1_9" or "2_Minor_10_25" or "3_Major_26_50" or "4_Destroyed_50plus",
        "Confidence": <float 0.0-1.0>
    }}
    """

    try:
        with open(image_path, "rb") as f:
            im_bytes = f.read()
        im = Image.open(io.BytesIO(im_bytes))
        im.thumbnail([1024, 1024])

        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[prompt, im],
            config=types.GenerateContentConfig(
                temperature=0.0, # Zero-shot classification requires absolute determinism
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                ]
            )
        )
        return json.loads(response.text)

    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
        return None

# --- 3. Batch Processing Logic ---

def process_classification_experiment(root_path):
    results = []

    # Your 5 folder names
    target_folders = ["0_No_Damage", "1_Affected_1_9", "2_Minor_10_25", "3_Major_26_50", "4_Destroyed_50plus"]

    print(f"Scanning dataset at: {root_path}")

    for folder_name in target_folders:
        folder_path = os.path.join(root_path, folder_name)

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_name}, skipping...")
            continue

        images = glob.glob(os.path.join(folder_path, "*.*"))
        # Compatible with more image formats
        images = [f for f in images if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        print(f"📂 Class [{folder_name}]: Found {len(images)} images.")

        for i, img_path in enumerate(images):
            print(f"   Processing {i+1}/{len(images)}: {os.path.basename(img_path)} ...", end="\r")

            prediction = classify_building_damage(img_path)

            if prediction:
                record = {
                    "Image_ID": os.path.basename(img_path),
                    "Ground_Truth_Class": folder_name,
                    "Predicted_Class": prediction.get("Predicted_Class", "Unknown"),
                    "Confidence": prediction.get("Confidence", 0),
                    # Determine if correct (1=Right, 0=Wrong)
                    "Is_Correct": 1 if prediction.get("Predicted_Class") == folder_name else 0
                }
                results.append(record)

                time.sleep(0.5)
        print("")

    return results

# --- 4. Execution and Saving ---

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT_PATH):
        records = process_classification_experiment(DATASET_ROOT_PATH)

        if records:
            df = pd.DataFrame(records)

            # Calculate accuracy
            accuracy = df["Is_Correct"].mean()
            print("\n" + "="*50)
            print(f"✅ Experiment Completed!")
            print(f"Total Images: {len(df)}")
            print(f"Overall Accuracy: {accuracy:.2%}")
            print(f"Results saved to: {OUTPUT_EXCEL_FILE}")
            print("="*50)

            df.to_excel(OUTPUT_EXCEL_FILE, index=False)

            try:
                from IPython.display import display
                display(df.head())
            except: pass
    else:
        print(f"❌ Error: Root path not found {DATASET_ROOT_PATH}")
