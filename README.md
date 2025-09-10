# ML_VISUAL_BACK

ML_VISUAL_BACK is a lightweight backend framework for machine learning experiments and visualization.  
It provides a unified interface for datasets, splitters, models, and evaluation, with support for both command-line and WebSocket-based execution.

## Features

- **Dataset support**: Built-in wrappers for common datasets (Iris, Wine, Breast Cancer, Diabetes, Boston Housing).  
- **Models included**: Decision Tree, Naive Bayes, K-Nearest Neighbors, Support Vector Machine, Random Forest, Logistic Regression, Linear Regression, Gradient Boosting, XGBoost, K-Means.  
- **Evaluation modules**: Metrics for classification (accuracy, kappa, hamming loss), regression (MSE, MAE, R²), and clustering (FMI, Calinski-Harabasz).  
- **Execution managers**:
  - `CmdManager`: run pipelines interactively from the command line.  
  - `WebManager`: expose pipelines over WebSocket, enabling integration with a front-end UI.  
- **Logging system**: Console and WebSocket loggers, supporting text and image output.  

## Project structure

