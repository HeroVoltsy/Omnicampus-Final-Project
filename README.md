# Omnicampus: The Final Project Dossier

**Investigating Feature Redundancy and Multicollinearity in Machine Learning-Based Customer Churn Prediction**

A machine learning research project conducted under the **Omnicampus Global Consumer Intelligence (GCI) Program**, examining how feature redundancy and multicollinearity influence telecommunications customer churn prediction.

## Overview

This study investigates whether reducing potentially redundant numerical predictors can preserve or improve the performance of machine learning models for customer churn classification.

Using approximately **100,000 telecommunications customer records and 49 numerical predictors**, the project examines feature dependencies through exploratory data analysis, correlation analysis, and singular value decomposition (SVD). Three feature configurations are then evaluated across six supervised classification algorithms under a consistent experimental framework.

Rather than assuming that fewer predictors necessarily produce better models, this research empirically evaluates the relationship between feature composition and predictive performance.

## Research Objectives

- Examine feature redundancy and multicollinearity among telecommunications customer attributes.
- Investigate numerical dependencies using correlation analysis, matrix rank, singular values, and condition numbers.
- Construct alternative feature configurations through controlled feature ablation.
- Compare six machine learning classifiers under consistent training and validation procedures.
- Evaluate whether reducing potentially redundant predictors preserves predictive performance and generalization.

## Experimental Design

### Dataset

- Approximately 100,000 customer records
- 49 numerical predictor variables
- Binary target: customer churn
- Customer ID excluded from the predictor set
- Median imputation incorporated into preprocessing pipelines

### Feature Configurations

| Configuration | Predictors | Description |
|---|---:|---|
| Feature Set A | 49 | Full feature set |
| Feature Set B | 47 | Aggregate-reduced feature set |
| Feature Set C | 44 | Dependency-reduced feature set |

### Machine Learning Models

1. Logistic Regression
2. Extra Trees
3. Random Forest
4. XGBoost
5. LightGBM
6. CatBoost

### Validation Strategy

- 80:20 stratified train-test split
- Five-fold stratified cross-validation
- Fixed random state: 42
- Pipeline-based preprocessing to prevent information leakage
- MLflow experiment tracking

### Evaluation Metrics

- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient (MCC)
- Sensitivity
- Specificity
- Brier Score

MCC serves as the primary classification comparison metric.

## Key Results

### Held-Out Test Performance

| Feature Set | Model | ROC-AUC | PR-AUC | MCC | Brier Score |
|---|---|---:|---:|---:|---:|
| A | XGBoost | 0.6467 | 0.6573 | 0.2024 | 0.2328 |
| B | LightGBM | 0.6473 | 0.6579 | **0.2030** | 0.2326 |
| C | LightGBM | **0.6477** | **0.6587** | 0.1986 | **0.2325** |

### Principal Findings

- Reducing potentially redundant predictors maintained predictive performance comparable to the complete feature set.
- Feature Set C with LightGBM achieved the highest held-out ROC-AUC and PR-AUC.
- Feature Set B with LightGBM achieved the highest held-out MCC.
- Feature reduction alone did not produce substantial improvements in classification quality.
- The results do not establish that one feature configuration is statistically superior to another.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost, LightGBM, CatBoost |
| Experiment Tracking | MLflow |
| Visualization | Matplotlib, Seaborn |
| Numerical Diagnostics | Correlation Analysis, Singular Value Decomposition |

Experiments were conducted in a local, primarily CPU-based computing environment with controlled execution to manage computational resources.

## Research Limitations

The study is limited to the supplied telecommunications dataset and its numerical predictors. It does not establish causal relationships, include external validation, or evaluate real-world deployment and customer retention interventions.

The reported results reflect the selected train-test partition and experimental procedure.

## Future Research Directions

Potential extensions include:

- Paired statistical comparisons and bootstrap confidence intervals
- Classification threshold optimization
- Probability calibration
- Systematic hyperparameter optimization
- Deep learning-based feature representation
- Hybrid and ensemble architectures
- External validation across independent telecommunications datasets

## Project Context

**Program:** Omnicampus Global Consumer Intelligence (GCI)  
**Research Domain:** Machine Learning, Telecommunications Analytics, Predictive Modeling  
**Research Focus:** Feature Redundancy, Multicollinearity, and Model Generalization  
**Author:** Jorge Creiann Jarme  
**Date:** November 2026

---

*This project emphasizes that reliable predictive modeling requires more than selecting the algorithm with the highest score. Understanding feature dependencies, model behavior, and generalization is equally important.*
