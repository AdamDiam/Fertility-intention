# Project Title: Interplay of Personal Characteristics and Social Networks in Shaping Fertility Intentions

## Overview
This project explores how personal characteristics and social network structures impact fertility intentions across different demographic groups. By employing advanced machine learning techniques, including XGBoost and Explainable Boosting Machine (EBM), this analysis quantifies the influence of both individual traits (e.g., age, income) and social network characteristics (e.g., density of kin, closeness to friends) on decisions regarding childbearing.

## Data
The analysis uses data from the LISS panel managed by Centerdata at Tilburg University, which includes comprehensive demographic and network information on individuals. The dataset focuses on a range of variables, including age, income, educational background, and detailed aspects of social networks.

## Methodology
The project applies a pipeline of preprocessing, feature selection, and modeling to understand the dynamics of fertility intentions. Key steps include:

- **Data Preprocessing**: Handling missing values and applying necessary transformations to prepare the data for modeling.
- **Feature Selection**: Utilizing Recursive Feature Elimination with Cross-Validation (RFECV) integrated with XGBoost to identify significant features.
- **Over-sampling**: Implementing SMOTE to address class imbalance, ensuring the model's robustness across different demographic groups.
- **Modeling**: Employing Explainable Boosting Machine (EBM) for its interpretability to better understand how each feature influences fertility intentions.

## Results
The models highlight the dual influence of personal characteristics and social networks, with age and number of children being pivotal across almost all groups analyzed. Additionally, the influence of social ties, such as the density of kin and friends who want children, plays a significant role, illustrating how individual fertility intentions are intertwined with social norms.
