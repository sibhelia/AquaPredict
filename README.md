# 💧 AquaPredict: Advanced Water Quality & Potability Analysis

## Motivation & Context

Access to safe drinking water is a fundamental human right and a vital component of effective health protection policies. As stated in global health research, investments in water supply and sanitation yield significant economic benefits by reducing healthcare costs and improving quality of life.

**AquaPredict** is designed to bridge the gap between complex chemical laboratory data and actionable safety insights. It uses Machine Learning to provide a reliable, fast, and transparent assessment of water potability based on global standards.

---

## Key Features (MVP)

- **Dual-Input Analysis:**
  - **Single-Sample Mode:** Interactive manual entry for immediate field analysis.
  - **Batch-Processing Mode:** Upload large CSV datasets for city-wide or regional water quality assessments.
- **WHO Standards Comparison:** Automatically compares input metrics (pH, Sulfates, etc.) against World Health Organization guidelines.
- **Explainable AI (XAI):** Visualizes the "Why" behind every prediction, highlighting which parameters contributed most to the safety risk.
- **Geographic Risk Mapping:** Generates heatmaps for batch data to identify high-risk water zones.
- **Professional Reporting:** Generates downloadable PDF reports including risk scores, parameter analysis, and safety recommendations.

---

## Technical Architecture

| Component | Technology |
| :--- | :--- |
| Model | Ensemble Learning (Random Forest / XGBoost) |
| Preprocessing | Advanced missing value handling (Iterative Imputation) + data scaling |
| Interface | Streamlit |
| Insights | SHAP / LIME for model transparency and interpretability |

---

## Dataset Parameters (Kaggle - Water Potability)

The model analyzes **9 critical metrics**:

| Parameter | Description |
| :--- | :--- |
| pH | Acid-base balance of the water. |
| Sulfate | Dissolved minerals affecting digestion. |
| Chloramines | Disinfectant levels used in public water. |
| Turbidity | Measure of light-emitting property (clarity). |
| Conductivity | Ion concentration and electrical conductivity. |
| Hardness | Calcium and magnesium mineral content. |
| Solids | Total dissolved solids in water. |
| Organic Carbon | Amount of organic compounds present. |
| Trihalomethanes | Byproducts formed during water disinfection. |

---

## Installation & Usage

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/aquapredict.git
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the application:**
```bash
streamlit run app.py
```

---

## Project Goal

This project demonstrates a complete end-to-end **Data Science lifecycle** — from advanced data imputation and imbalanced data handling to deploying a transparent, high-value software product that addresses real-world environmental challenges.
