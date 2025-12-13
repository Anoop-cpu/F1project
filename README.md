# 🏎️ Formula 1 Podium Prediction Using Machine Learning

This repository contains the complete implementation, experiments, and evaluation
for the project **“Predicting Formula 1 Podium Finishes Using Ensemble Machine Learning Models.”**
The project focuses on predicting whether a driver will finish on the podium
(top 3) in a Formula 1 race using historical race data and ensemble learning methods.

The work accompanies an IEEE-style research paper and includes data preprocessing,
model training, evaluation, and a lightweight interactive application for inference
and visualization.

---

## 📌 Project Overview

Predicting Formula 1 race outcomes is challenging due to:
- Severe **class imbalance** (only 3 podium finishers per race)
- **Non-stationarity** caused by regulation changes and driver transfers
- Complex interactions between qualifying position, driver form, team strength, and track context

This project formulates podium prediction as a **binary classification and ranking problem**
and evaluates two ensemble models:
- **XGBoost**
- **Random Forest**

Evaluation emphasizes **ranking-aware metrics** such as ROC-AUC, PR-AUC, Lift curves,
and a race-level **Top-3 Hit Rate**, rather than accuracy alone.

---

## 📊 Dataset

- Source: **Kaggle Formula 1 World Championship Dataset**
- Seasons used: **2014–2024**
  - Data prior to 2014 was removed to ensure consistency with the FIA V6 hybrid era
- Unit of analysis: **(Driver, Race)** pairs
- Target variable:
  - `1` → Podium finish
  - `0` → Non-podium finish

Rookie drivers and unseen seasons were handled carefully to avoid data leakage
and preserve real-world generalization.

---

## 🧠 Models Implemented

- **Random Forest**
  - Robust baseline
  - Strong recall for podium finishes with deeper trees
- **XGBoost**
  - Gradient-boosted trees
  - Superior ranking performance with fewer estimators
  - Computationally efficient

Hyperparameter sensitivity experiments were conducted for:
- Number of estimators
- Maximum tree depth

---

## 📈 Evaluation Metrics

The following metrics are reported:
- Accuracy
- ROC-AUC
- PR-AUC (Average Precision)
- Lift Curve
- Confusion Matrix
- **Top-3 Hit Rate per race** (race-level evaluation)

These metrics are visualized using plots included in the paper and repository.

---

## 📂 Repository Structure
 - app.py                     # Streamlit app for interactive prediction
 - F1.py                      # Model training and evaluation logic
 - features.py                # Feature engineering and preprocessing
 - Testing_2025.py            # Testing on unseen season data
 - figures/                   # Generated plots (ROC, PR, Lift, Metrics)
 - data/                      # Processed CSV files (not all included)
 - Ieee_Report                # IEEE LaTeX source files
 - README.md                  # Project documentation

---

## ▶️ Running the Streamlit Application

The project includes an interactive **Streamlit** app for running predictions
and visualizing results.

### 🔹 Prerequisites
Make sure you have Streamlit installed:
~~~sh
pip install streamlit
streamlit run /Location/on/your/app.py
~~~
Replace /Location/on/your/app.py with the actual path to app.py on your system.
Example
~~~sh
streamlit run ~/projects/f1-podium-prediction/app.py
~~~

---

## 📄 Research Paper
An IEEE-format research paper describing the methodology, experiments,
and results is included in this repository.
The paper covers:
	•	Dataset construction and preprocessing
	•	Model comparison
	•	Hyperparameter sensitivity analysis
	•	Ranking-based evaluation
  
---

## 🔗 Project Link
The full source code and experiments are available in this repository.
If you use this work, please cite the accompanying paper or reference this repository.

---

## 👤 Author
Anoop Lashiyal
M.S. Computer Science
Georgia State University

---

### 📜 License
This project is intended for academic and educational use.
