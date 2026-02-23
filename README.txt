# PROJECT TITLE: Gender-Based Comparative Analysis of Heart Disease Prediction Models

## GROUP MEMBERS: Muhammad Umair Akram Butt (mubu25@student.bth.se)
## COURSE: DV2638 - Machine Learning
## INSTITUTION: Blekinge Institute of Technology (BTH)
## DATE: 9 January 2026

# 1. PROJECT OVERVIEW

This project implements a comprehensive two-stage machine learning analysis to compare heart disease prediction models across genders. The study employs three algorithms (Logistic Regression, Random Forest, XGBoost) and investigates whether gender-specific models outperform unified approaches.


# 2. Code and Dataset Files

- `heart_disease_analysis.ipynb`: The main Jupyter Notebook containing the code, analysis, and results.
- `Dataset Old/heart+disease/`: Directory containing the dataset files (`*.data`).
- requirements.txt: Text file containing the dependencies information.

# 3. Prerequisites

The code requires Python 3.8 or higher and the following libraries (as described in the requirements.txt):
- pandas==2.3.1
- numpy==1.26.4
- matplotlib==3.9.2
- seaborn==0.13.2
- scikit-learn==1.5.1
- xgboost==3.1.2
- imbalanced-learn==0.12.3
- joblib==1.4.2
- scipy==1.15.2

# 3. Code Execution Instructions

Open the terminal at the root directory, and run this command to install all dependencies:

```
pip install -r requirements.txt
```

Then, launch the Jupyter notebook using this command:

```
jupyter notebook heart_disease_analysis.ipynb
```

Then, run all the cells.


# 4. Dataset Information

Dataset Source: UCI Heart Disease Database [https://archive.ics.uci.edu/dataset/45/heart+disease]
Location: Dataset Old/heart+disease/

Files Used:
- processed.cleveland.data (303 samples)
- processed.hungarian.data (294 samples)
- processed.switzerland.data (123 samples)
- processed.va.data (200 samples)


# 5. File STructure

Project Root/
│
├── heart_disease_analysis.ipynb    # Main Jupyter notebook (main code file)
├── requirements.txt                # Python dependencies
├── README.txt                      # This file
│
├── Dataset Old/                    # Original UCI datasets directory
│   └── heart+disease/
│       ├── processed.cleveland.data
│       ├── processed.hungarian.data
│       ├── processed.switzerland.data
│       └── processed.va.data
│
├── images/                         # Generated graphs
│   ├── target_distribution.png
│   ├── target_distribution_pie_chart.png
│   ├── gender_distribution_plots.png
│   ├── feature_correlation_plot.png
│   ├── feature_distribution_by_gender.png
│   ├── model_comparison_for_male.png
│   ├── model_comparison_for_female.png
│   ├── Final_coparison_Plots.png
│   └── Final_feature_importance_comparison.png
│
└── models/                         # Saved trained models
   ├── stage1_best_male_random_forest.pkl
   ├── stage1_best_female_logistic_regression.pkl
   ├── stage2_male_oversampled_random_forest.pkl
   ├── stage2_female_oversampled_logistic_regression.pkl
   ├── stage2_male_undersampled_random_forest.pkl
   ├── stage2_female_undersampled_logistic_regression.pkl
   └── scaler_*.pkl (various scalers)