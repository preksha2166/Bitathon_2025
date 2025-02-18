# Bitathon_2025
# Financial Distress Analysis

## Overview
This project analyzes financial distress among individuals using data processing, visualization, and machine learning techniques. It involves cleaning financial data, generating insights through visualizations, and predicting financial distress using various machine learning models, including Random Forest and Neural Networks.

## Features
- Data preprocessing and cleaning using Pandas & NumPy
- Exploratory Data Analysis (EDA) with Seaborn and Matplotlib
- Financial distress prediction using:
  - K-Means Clustering
  - Random Forest Classifier
  - Neural Networks (Keras & TensorFlow)
- Feature Engineering and Hyperparameter Tuning
- Model Evaluation using classification metrics

## Requirements
To run this project, install the following dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow
```

## Dataset
The dataset contains demographic and financial indicators, including:
- `age`, `educ`, `inc_q` (Income Quintile)
- `account`, `saved`, `borrowed` (Financial behavior)
- `financial_distress` (Target variable for classification)

## Usage
1. Load the dataset:
   ```python
   import pandas as pd
   df = pd.read_csv('data.csv')
   ```
2. Preprocess the data:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   df_scaled = scaler.fit_transform(df.drop('financial_distress', axis=1))
   ```
3. Train a Random Forest model:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['financial_distress'], test_size=0.2, random_state=42)
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   ```
4. Evaluate the model:
   ```python
   from sklearn.metrics import accuracy_score, classification_report
   y_pred = model.predict(X_test)
   print(accuracy_score(y_test, y_pred))
   print(classification_report(y_test, y_pred))
   ```

## Visualizations
- Age distribution
- Financial distress correlation heatmap
- Savings and borrowing trends by gender

## Future Improvements
- Implement additional deep learning models
- Improve data balancing for better model performance
- Deploy the model as a web application

## Contributors
- **Your Name**

## License
This project is open-source under the MIT License.

