# Smart-grid-Optimization

## Overview

This repository provides an end-to-end solution for forecasting energy consumption in smart grids using machine learning techniques. The project is designed to analyze and optimize energy usage by leveraging regression models to predict future consumption based on historical data and time-based features.

## Features

- **Upload and preprocess datasets:** Easily upload a productivity or energy consumption dataset (CSV format) and perform comprehensive preprocessing, including:
  - Parsing and extracting features from datetime columns (Month, Hour, Day, Year, Minute)
  - Outlier handling using Winsorization
  - Feature selection using SelectKBest
  - Data scaling with StandardScaler

- **Model Training and Evaluation:**
  - Train and evaluate regression models such as Support Vector Regression (SVR) and Random Forest Regressor (RFR).
  - Calculate and display key performance metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.
  - Visualize model performance with scatter plots comparing actual and predicted values.

- **Prediction:**
  - Load new test datasets for prediction.
  - Output predicted energy consumption values for each record in the test set.

- **Interactive GUI:**
  - Tkinter-based user interface for easy navigation, data upload, preprocessing, model training, evaluation, and prediction.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/satwikasunkari/Smart-grid-Optimization.git
   cd Smart-grid-Optimization
   ```

2. **Install dependencies:**
   Make sure you have Python 3.x and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```
   The GUI will launch, allowing you to upload your dataset, preprocess the data, train models, and make predictions.

## Dataset

- The dataset should be a CSV file with at least a `DateTime` column and a `TotalUsage` column (target variable).
- The preprocessing step extracts additional time features for model input.

## Machine Learning Models

- **Support Vector Regressor (SVR):**
  - Used for regression tasks to predict energy consumption.
  - Model persistence is handled with joblib for reuse.

- **Random Forest Regressor:**
  - Ensemble regression model to capture complex relationships.
  - Also persisted with joblib.

## Performance Metrics

After training, the following metrics are calculated and displayed:
- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

## Visualization

- Box plots for each feature to visualize and handle outliers.
- Scatter plots comparing actual versus predicted values.

## Dependencies

Key Python libraries used include:
- pandas, numpy, matplotlib, seaborn
- scikit-learn (for ML algorithms and preprocessing)
- imbalanced-learn (SMOTE for oversampling)
- feature_engine (for outlier handling)
- joblib (for model persistence)
- tkinter (for GUI)

## License

This project is licensed under the MIT License.

---

**Author**: [satwikasunkari](https://github.com/satwikasunkari)
