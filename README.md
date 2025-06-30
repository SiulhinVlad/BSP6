# Fine-Tuning an LLM with Federated Learning for Multimodal Radiology Report Summarization

This repository contains code and results for fine-tuning transformer models (T5 and BART) to automatically summarize chest X-ray radiology reports, in three stages:

1. **Text-only baseline** (centralized fine-tuning on paired “findings” → “impression”).  
2. **Centralized multimodal** (integrating chest X-ray image features via a ViT→T5/BART wrapper).  
3. **Federated multimodal** (simulating FedAvg across **7** clients, preserving data locality).

_Report Overleaf_: https://www.overleaf.com/read/krvdbspsdggv#39426d
