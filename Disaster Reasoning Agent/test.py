import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Any
from google import genai
from google.genai import types
from IPython.display import display, HTML

# Ensure API Key is set in the environment variables
# os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"

class DisasterReasoningAgent:
    """
    Implements the Semantic Reasoning Module of the GeoAgent4Disaster framework.
    
    This agent aligns visual perception data (from RSI and SVI) with domain knowledge 
    to generate spatially explicit, explainable situational assessment reports 
    in a zero-shot manner.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the reasoning engine.

        Args:
            api_key (str): Authentication key for the LLM backend.
            model_name (str): The specific model version for inference.
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _serialize_visual_evidence(self, observation_df: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Serializes the structured visual detection data into a natural language context 
        for the Large Multimodal Model (LMM).

        Args:
            observation_df (pd.DataFrame): Dataframe containing detected objects, 
                                           source modality (RSI/SVI), and damage levels.

        Returns:
            Tuple[str, List[str]]: A tuple containing the serialized text context 
                                   and a list of unique detected object categories.
        """
        context_lines = []
        unique_objects = set()
        
        for _, row in observation_df.iterrows():
            obj_category = row.get('Object', 'Unidentified')
            severity = row.get('Level', 'Unknown')
            modality = row.get('Source', 'Multimodal')
            description = row.get('Context', 'No contextual details provided')
            
            unique_objects.add(obj_category)
            
            # Formulate a structured observation log
            entry = (
                f"- **Modality**: {modality}\n"
                f"  - Entity: {obj_category}\n"
                f"  - Damage State: {severity}\n"
                f"  - Semantic Context: {description}\n"
            )
            context_lines.append(entry)
            
        return "\n".join(context_lines), list(unique_objects)

    def execute_inference(self, observation_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes the Chain-of-Thought (CoT) reasoning process to generate the disaster report.

        This method constructs the prompt, invokes the LMM, and parses the JSON output.

        Args:
            observation_df (pd.DataFrame): The input visual perception data.

        Returns:
            Dict[str, Any]: The structured JSON assessment report.
        """
        
        # 1. Data Serialization
        evidence_context, detected_entities = self._serialize_visual_evidence(observation_df)
        entities_str = ", ".join(detected_entities)
        
        # 2. Prompt Engineering
        # Designed to enforce the output schema required for the comparative study table.
        system_prompt = f"""
        Role: Autonomous Disaster Response Strategist (GeoAgent).
        
        Task: Synthesize a "Situational Awareness Report" based on the provided multi-view visual evidence. 
        The evidence combines Remote Sensing Imagery (RSI) for macro-scale context and Street View Imagery (SVI) for micro-scale details.

        === VISUAL PERCEPTION LOG ===
        {evidence_context}

        === INFERENCE REQUIREMENTS ===
        1. **Cross-View Validation**: Integrate findings from both RSI (e.g., saturated ground) and SVI (e.g., debris piles) to validate the disaster type.
        2. **Causal Reasoning**: Explain the link between the visual cues (e.g., gutted homes) and the physical mechanism (e.g., storm surge).
        3. **Actionability**: Propose recovery steps based on the functional criticality of the damaged infrastructure.

        === OUTPUT SCHEMA (JSON) ===
        Generate a valid JSON object with the following keys exactly:
        {{
            "Type of disaster": "String (e.g., Hurricane, Earthquake)",
            "Image restoration": "String (No/Yes)",
            "Disaster severity classification": "String (Severe/Moderate/Minor)",
            "Voted candidate": "String (e.g., SVI and RSI)",
            "Confidence": "Float (0.0 - 1.0)",
            "Object recognition": "String (List key entities detected)",
            "Disaster Assessment Suggestion Reasoning": "String (Detailed analytical paragraph explaining the severity)",
            "Disaster Recovery / Reconstruction Suggestion Reasoning": "String (Detailed actionable paragraph for response teams)"
        }}

        Ensure the reasoning is logically sound and consistent with the detected entities: {entities_str}.
        """

        print(f"[{self.__class__.__name__}] Executing zero-shot inference on {self.model_name}...")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=system_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1  # Low temperature for deterministic output
                )
            )
            return json.loads(response.text)
        except Exception as e:
            return {
                "Error": "Inference Failure",
                "Details": str(e), 
                "Type of disaster": "Unknown"
            }

    def render_report_table(self, report_json: Dict[str, Any]) -> pd.DataFrame:
        """
        Transforms the JSON report into a formatted DataFrame for visualization,
        matching the evaluation metrics structure.
        """
        # Define the schema order for presentation
        schema_order = [
            "Type of disaster",
            "Image restoration",
            "Disaster severity classification",
            "Voted candidate",
            "Confidence",
            "Object recognition",
            "Disaster Assessment Suggestion Reasoning",
            "Disaster Recovery / Reconstruction Suggestion Reasoning"
        ]
        
        # Extract and align data
        data_map = {
            "GeoAgent (Proposed)": [report_json.get(k, "N/A") for k in schema_order]
        }
        
        df_result = pd.DataFrame(data_map, index=schema_order)
        df_result.index.name = "Evaluation Metrics"
        
        return df_result

# ==========================================
# Main Execution Block (Simulation)
# ==========================================

if __name__ == "__main__":
    # 1. Synthesize Perception Data
    # Simulates the output from the upstream 'Perception Agent' (e.g., object detection results)
    # mirroring the specific Hurricane Ian scenario shown in the reference image.
    perception_data = [
        {
            "Source": "SVI", 
            "Object": "Debris (Furniture/Appliances)", 
            "Level": "Severe", 
            "Context": "Massive piles obstructing the roadway, indicating structural gutting."
        },
        {
            "Source": "SVI", 
            "Object": "Vegetation", 
            "Level": "Moderate", 
            "Context": "Trees stripped of leaves but remaining upright."
        },
        {
            "Source": "SVI", 
            "Object": "Residential Structure", 
            "Level": "Severe", 
            "Context": "Homes appear gutted by hydraulic force (surge)."
        },
        {
            "Source": "RSI", 
            "Object": "Terrain/Soil", 
            "Level": "Severe", 
            "Context": "High water saturation visible in aerial view."
        }
    ]
    
    df_perception = pd.DataFrame(perception_data)

    # 2. Initialize Agent
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable not found.")
    else:
        reasoning_agent = DisasterReasoningAgent(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name="gemini-1.5-pro" 
        )

        # 3. Run Inference
        assessment_report = reasoning_agent.execute_inference(df_perception)

        # 4. Visualization
        # Renders the report as an HTML table suitable for Jupyter Notebooks
        df_display = reasoning_agent.render_report_table(assessment_report)
        
        # Configure Pandas for full text display (essential for the Reasoning fields)
        pd.set_option('display.max_colwidth', None)
        
        # Display formatted output
        display(HTML(df_display.to_html().replace("\\n", "<br>")))
