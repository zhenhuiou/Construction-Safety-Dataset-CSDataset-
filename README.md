# Construction Safety Dataset and Benchmark (CIKM 2025 Resource Track)

This repository contains the **Construction Safety Dataset** and the codebase accompanying our paper accepted to the **CIKM 2025 Resource Track**. This work introduces a large-scale, multi-level, multi-modal dataset for construction safety research, and provides strong experimental benchmarks for both **predictive** and **causal** modeling tasks.

---

## 🏗️ Dataset Overview

Our dataset integrates three critical levels of construction safety data, collected from OSHA's official records:

- **Incident-level**: Records of construction-related injury/fatality incidents with severity, location, and narrative description.
- **Inspection-level**: Regulatory inspection metadata, types (e.g., complaint-driven), and timestamps.
- **Violation-level**: Detailed citations, OSHA standards violated, and violation summaries.

Each record is aligned using unique IDs like `activity_nr` and `violator_id`, enabling cross-level linkage and temporal analysis.

The dataset combines:

- **Tabular features**: weather conditions, job titles, inspection type, time/location
- **Textual features**: incident abstracts, violation descriptions

---

## 🔬 Tasks and Experiments

We evaluate the dataset on two key tasks to demonstrate its benchmarking potential:

### 4.1 Injury Severity Prediction

**Objective**: Predict the severity (1 to 4) of an injury using structured and unstructured features.

#### Models Evaluated:

- Traditional ML: Logistic Regression (L1/L2), SVM, Random Forest, XGBoost, MLP
- LLMs: ChatGPT-4.1mini, Qwen2.5-7B

#### Results:

| Model            | Accuracy | Macro-F1 |
|------------------|----------|----------|
| Random Forest    | 0.8239   | 0.8132   |
| MLP              | 0.8220   | 0.8114   |
| XGBoost          | 0.8190   | 0.8089   |
| ChatGPT-4.1mini  | 0.8350   | 0.8200   |
| Qwen2.5-7B       | 0.8300   | 0.8183   |

LLMs outperformed traditional models due to their ability to leverage narrative context and model cross-level relationships between violations, inspections, and incidents.

---

### 4.2 Inspection-Incident Causal Analysis

**Objective**: Evaluate the causal impact of complaint-driven inspections on subsequent incident occurrence.

We conducted a **quasi-experimental analysis** using **propensity score matching (PSM)**:

1. **Identify treatment group**: inspections triggered by complaints.
2. **Identify control group**: inspections triggered by other reasons.
3. **Match cohorts**: using PSM on covariates such as industry, region, firm size, and prior inspection count.
4. **Compare outcomes**: measure difference in incident rates after inspections.

#### Key Result:
> Complaint-driven inspections reduce the likelihood of downstream incidents by **9.8%**.

This experiment showcases the dataset's ability to support **causal modeling**, **temporal inference**, and **policy evaluation** through rich cross-level linkages.
