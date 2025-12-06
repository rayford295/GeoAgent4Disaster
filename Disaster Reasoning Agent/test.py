import pandas as pd
import json
import os
from google import genai
from google.genai import types
from IPython.display import Markdown, display

class DisasterReasoningAgent:
    """
    3.2.4 Disaster Reasoning Agent (No-GIS Version)
    角色：基于视觉语义的灾害链推理者
    功能：仅依赖视觉识别结果，利用 LLM 的常识库进行因果分析和策略生成。
    """

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def _format_visual_evidence(self, damage_df: pd.DataFrame) -> str:
        """
        数据预处理
        功能：将结构化的 DataFrame 转换为 LLM 易读的自然语言描述列表。
        不再需要 GIS 融合，而是强调“对象类型”和“损伤等级”。
        """
        evidence_list = []
        
        # 遍历损伤表格
        for _, row in damage_df.iterrows():
            obj_id = row['ID']
            obj_type = row['Object']
            damage_type = row['Damage_Type']
            level = row['Level']
            # reasoning = row.get('Reasoning', 'N/A') # 如果上一主要有推理字段也可以加上
            
            # 构建描述字符串
            # 强调对象的功能属性隐含在 Object 名称中 (如 'Power Lines' implies Infrastructure)
            record = (
                f"- **Object ID**: {obj_id}\n"
                f"  - Category: {obj_type}\n"
                f"  - Damage State: {damage_type} (Severity: {level})\n"
            )
            evidence_list.append(record)
            
        return "\n".join(evidence_list)

    def run_reasoning_engine(self, damage_df: pd.DataFrame):
        """
        阶段 B & C: Disaster Chain Reasoning & Recovery Strategist
        """
        
        # 1. 格式化视觉证据
        print("--- Step 1: Aggregating Visual Evidence ---")
        evidence_text = self._format_visual_evidence(damage_df)
        
        # 2. 构建推理提示词 (Chain-of-Thought)
        # 这里的 Prompt 经过调整，让 AI 利用通用知识 (Common Sense) 来替代 GIS 数据
        
        prompt = f"""
        You are a Chief Disaster Response Strategist. 
        Analyze the following "Visual Damage Inspection Report" derived from street-view imagery of a hurricane-hit area.
        
        Since specific GIS location data is unavailable, you must rely on **Object Semantics** (the nature of the object) to determine priority.

        === VISUAL EVIDENCE INPUT ===
        {evidence_text}
        
        === REASONING TASKS ===
        Perform the analysis in three logical stages:

        Stage 1: Semantic Causal Inference (The "Disaster Chain")
        - **Attribution**: Based on the object and damage type, infer the cause (e.g., "Tree uprooted" -> likely high wind gusts; "Flooded street" -> likely storm surge or heavy rain).
        - **Propagation & Interaction**: Infer how one damage might affect another. 
          *Example: If there are 'downed power lines' AND 'flooded streets', identify the extreme electrocution risk.*
          *Example: If 'debris on road', infer that emergency access is blocked.*

        Stage 2: Situational Assessment
        - Provide an overall "Disaster Severity Score" (1-10) for this scene.
        - Describe the "Immediate Threats" (e.g., live wires, blocked escape routes).

        Stage 3: Recovery Strategy (Action Plan)
        - Generate a prioritized repair list based on **Functional Criticality**: 
          **Rule**: Life-Safety Hazards (Power/Collapse) > Mobility (Roads) > Property (Roofs/Facades) > Aesthetics (Trees).
        - Provide specific resource commands (e.g., "Dispatch utility crew for ID_xx", "Send bulldozer for ID_yy").

        === OUTPUT FORMAT ===
        Output a structured Markdown report titled "## Context-Aware Disaster Assessment Report".
        """

        print("--- Step 2: Running Reasoning Engine (Gemini 1.5 Pro) ---")
        try:
            # 使用 1.5 Pro 进行逻辑推理
            response = self.client.models.generate_content(
                model="gemini-1.5-pro",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3)
            )
            return response.text
        except Exception as e:
            return f"Reasoning Engine Failed: {e}"

# ==========================================
# 模拟运行 (Simulation)
# ==========================================

# 1. 准备数据：假设这是上一部 Agent (DamageRecognitionAgent) 输出的真实 DataFrame
# 注意：这里完全没有 GIS ID，只有识别出的物体本身
mock_damage_data = {
    "ID": ["obj_0", "obj_1", "obj_2", "obj_3", "obj_4"],
    "Object": [
        "collapsed structure",      # 倒塌结构 (隐含：生命危险)
        "fallen tree blocking road",# 倒树 (隐含：交通阻断)
        "damaged roof",             # 屋顶 (隐含：财产损失)
        "downed power lines",       # 电线 (隐含：极度危险)
        "flooded street"            # 积水 (隐含：阻碍通行)
    ],
    "Damage_Type": ["Structural Failure", "Blockage", "Roof Loss", "Electrical Hazard", "Flooding"],
    "Level": ["Severe", "Moderate", "Moderate", "Severe", "Moderate"],
    # Location 数据在这一步主要用于展示，推理主要靠 Object 语义
    "Location": [[0,0,0,0]] * 5 
}
df_agent3_output = pd.DataFrame(mock_damage_data)

# 2. 初始化 Agent
reasoning_agent = DisasterReasoningAgent(api_key=os.environ["GEMINI_API_KEY"])

# 3. 运行推理 (不再传入 geo_context)
final_report = reasoning_agent.run_reasoning_engine(df_agent3_output)

# 4. 显示报告
display(Markdown(final_report))
