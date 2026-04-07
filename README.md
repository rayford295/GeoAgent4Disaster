# Agent4Disaster

**Towards Autonomous Disaster Assessment: A Cross-View Multi-Agent Pipeline for Zero-Shot Damage Diagnosis**

---

## Overview

Agent4Disaster is an autonomous **GeoAI multi-agent framework** for hyperlocal, interpretable, and near-real-time disaster assessment. It integrates multimodal geospatial inputs — satellite imagery (RSI), street-view imagery (SVI), and temporal change information — into an end-to-end pipeline covering disaster perception, image restoration, damage recognition, and recovery reasoning.

By leveraging **vision-language foundation models** and **agent-based orchestration**, the framework enables cross-view disaster understanding and zero-shot damage diagnosis without task-specific retraining.

<p align="center">
  <img src="https://github.com/rayford295/GeoAgent4Disaster/blob/main/figure/proposed%20framework.drawio.png" width="85%"/>
</p>

---

## Repository Structure

```
Agent4Disaster/
├── Disaster Perception Agent/
│   ├── DisasterPerceptionAgent.py
│   └── Prompt--Disaster Perception Agent
├── Image Restoration Agent/
│   ├── test.py
│   └── Prompt--Image Restoration Agent
├── Damage Recognition Agent/
│   ├── SVI&RSI.py
│   ├── SVI-pre&post.py
│   ├── SVI-wildfire.py
│   ├── zero_shot_object_detection_Agent3.ipynb
│   └── Prompt--Damage Recognition Agent
├── Disaster Reasoning Agent/
│   ├── Large Language Model-based evaluation.py
│   ├── test.py
│   └── Prompt--Disaster Reasoning Agent
└── figure/
```

---

## Pipeline

| Agent | Role |
|---|---|
| **Perception Agent** | Identifies disaster type, image modality, and structural context; plans downstream workflow |
| **Restoration Agent** | Enhances degraded SVI/RSI to improve structural visibility for downstream reasoning |
| **Recognition Agent** | Object-level damage detection, severity classification, and bi-temporal change reasoning |
| **Reasoning Agent** | Synthesizes outputs into structured reports with causal explanations and recovery recommendations |

---

## Example Outputs

| LLM-Based Object Detection | Final Structured Output |
|:---:|:---:|
| <img src="https://github.com/rayford295/GeoAgent4Disaster/blob/main/figure/example-llm-object%20detection.drawio.png" width="340"/> | <img src="https://github.com/rayford295/GeoAgent4Disaster/blob/main/figure/final%20output.png" width="340"/> |

<p align="center">
  <img src="https://github.com/rayford295/Agent4Disaster/blob/main/figure/reasoning_RESULTs.drawio.png" width="80%"/>
</p>

---

## Datasets

Evaluated across three multimodal disaster dataset categories:

- **Cross-view hurricane imagery** — paired SVI + RSI
- **Bi-temporal street-view imagery** — pre/post disaster pairs
- **Multi-hazard street-view datasets** — wildfire, flooding, earthquake

| Geolocation Distribution | Dataset Statistics |
|:---:|:---:|
| <img src="https://github.com/rayford295/GeoAgent4Disaster/blob/main/figure/geolocation.png" width="340"/> | <img src="https://github.com/rayford295/GeoAgent4Disaster/blob/main/figure/stastics.png" width="340"/> |

---

## Citation

```bibtex
@article{yang2026agent4disaster,
  title  = {Towards Autonomous Disaster Assessment: A Cross-View Multi-Agent
            Pipeline for Zero-Shot Damage Diagnosis},
  author = {Yang, Yifan and others},
  year   = {2026}
}
```

---

## Contact

**Yifan Yang** — Department of Geography, Texas A&M University
[yyang295@tamu.edu](mailto:yyang295@tamu.edu) · [rayford295.github.io](https://rayford295.github.io)
