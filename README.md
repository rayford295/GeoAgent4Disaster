# GeoAgent4Disaster
A Multi-Agent Framework for Multimodal Disaster Damage Assessment

<p align="center">
  <img src="https://github.com/rayford295/GeoAgent4Disaster/blob/main/figure/proposed%20framework.drawio.png" width="80%" />
</p>

## Overview
GeoAgent4Disaster is an autonomous multi-agent GeoAI framework designed for hyperlocal, interpretable, and near–real-time disaster assessment.  
The system integrates multimodal inputs—satellite imagery, street-view imagery, textual cues, and temporal change information—and performs a full pipeline of:

- Disaster perception  
- Image restoration  
- Damage recognition  
- Disaster reasoning & recovery recommendation  

By leveraging vision–language foundation models and agent-based orchestration, the framework enables cross-view understanding, zero/few-shot disaster analysis, and automated report generation without task-specific retraining.

This repository hosts the project materials associated with our research paper.

> **Note:** The full source code is currently **not publicly available** because the paper has not yet been released on a public platform.

---

## Key Features
- Multi-agent collaboration for perception, enhancement, recognition, and reasoning  
- Multimodal disaster interpretation (RSI + SVI + text)  
- Zero-shot cross-view disaster severity assessment  
- Structured JSON outputs for downstream analytics  
- Automated disaster situation reporting for the “golden 36 hours”  
- Evaluation across cross-view, bi-temporal, and multi-hazard datasets  

---

## Project Architecture
The GeoAgent4Disaster pipeline consists of four core agents:

### 1. Disaster Perception Agent  
Detects hazard type, identifies image mode, and plans downstream processing.

### 2. Image Restoration Agent  
Enhances degraded SVI/RSI for better structural clarity and reliable analysis.
<p align="center">
  <img src="https://github.com/rayford295/GeoAgent4Disaster/blob/main/figure/agent2_sample.png" width="70%" />
</p>

### 3. Damage Recognition Agent  
Performs object-level detection, severity classification, and change-based reasoning.

### 4. Disaster Reasoning Agent  
Synthesizes structured outputs to generate high-level causal interpretation and actionable recovery recommendations.

---

## Example Outputs

### LLM-Based Object Detection  
<p align="center">
  <img src="https://github.com/rayford295/GeoAgent4Disaster/blob/main/figure/example-llm-object%20detection.drawio.png" width="75%" />
</p>

*Figure 2. Example of object-level detection using vision–language models.*

---

### Final Output (Structured JSON + Explanation)

<p align="center">
  <img src="https://github.com/rayford295/GeoAgent4Disaster/blob/main/figure/final%20output.png" width="70%" />
</p>

*Figure 3. Final agent-produced output including severity, object detection, and reasoning.*

---

## Datasets
The framework supports and evaluates multiple multimodal disaster datasets, including:

- Cross-view hurricane imagery (paired SVI + RSI)  
- Bi-temporal street-view imagery (pre- vs. post-disaster)  
- Multi-hazard SVI datasets (wildfire, flooding, earthquake, etc.)

Dataset details and preprocessing scripts will be released alongside the paper.
<p align="center">
  <img src="https://github.com/rayford295/GeoAgent4Disaster/blob/main/figure/geolocation.png" width="80%" />
</p>

<p align="center">
  <img src="https://github.com/rayford295/GeoAgent4Disaster/blob/main/figure/stastics.png" width="70%" />
</p>

---
