# ML_VISUAL_BACK

## What does the project do?

ML_VISUAL_BACK is a backend framework designed to support machine learning experiments and visualization. It provides a unified interface for handling datasets, splitting data, training and testing models, and evaluating results. The framework includes implementations of classical algorithms such as decision trees, linear regression, Naive Bayes, k-nearest neighbors, support vector machines, gradient boosting, and XGBoost. Evaluation modules are built in for classification, regression, and clustering tasks.  

In addition to command-line execution, ML_VISUAL_BACK offers a WebSocket-based manager, enabling real-time communication with a front-end client. Through this integration, results such as metrics, logs, and plots can be displayed interactively in a visualization interface.

## Why is the project useful?

Machine learning models are often difficult to interpret directly. ML_VISUAL_BACK provides a structured way to set up experiments and makes results accessible through standardized evaluation and visualization. Researchers and developers can quickly run experiments, test multiple algorithms on well-known datasets, and view results in a consistent format.  

The WebSocket interface allows the backend to integrate seamlessly with a front-end system, making it possible to build interactive visualization tools for machine learning workflows. This design enables ML_VISUAL_BACK to serve both as an educational resource for understanding algorithms and as a practical backend for small-scale machine learning applications.

## How can users get started with the project?

To get started with ML_VISUAL_BACK, clone the repository and install the required dependencies:

```bash
git clone https://github.com/runrunaway2020/ML_VISUAL_BACK.git
cd ML_VISUAL_BACK
pip install numpy scikit-learn matplotlib xgboost websockets
