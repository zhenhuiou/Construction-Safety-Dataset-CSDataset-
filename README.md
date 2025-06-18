## 🔍 Overview

This repository contains the dataset and baseline code for the paper:

> **"Building Safer Sites: A Large-Scale Multi-Level Dataset for Construction Safety Benchmark"**  
> Accepted at **CIKM 2025 Resource Track**

We introduce a large-scale, multi-level dataset for construction safety analysis, integrating structured data (e.g., weather, location, job title, inspection records) and unstructured narratives (e.g., incident abstracts and violation summaries). The dataset is designed to support multi-modal modeling across three levels:

- **Incident**
- **Inspection**
- **Violation**

---

## Injury Severity Prediction

The task is to predict the **severity level** of each construction incident (1–4), based on both structured tabular features and unstructured text descriptions.

### 📦 Features Used:
- Weather: temperature, humidity, etc.
- Site metadata: city, state, zip code, job title
- Date features: year, month
- Narrative abstract: embedded using `sentence-transformers`

### ✅ Baseline Models:
We implemented and compared the following models:
- Logistic Regression (L1 & L2)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Multi-layer Perceptron (MLP)
- Large Language Models (ChatGPT, Qwen2.5)
