# Boston Housing Prediction Using Linear Regression

## Overview
This repository contains a detailed analysis of the **Boston Housing Dataset** using **Linear Regression**. The goal is to predict the **median value of homes (`medv`)** based on various features such as the number of rooms, crime rate, property tax rate, etc.

The workflow includes loading the dataset, preprocessing the data, building a linear regression model, evaluating its performance, and interpreting the results.

## Dataset
The dataset used in this project is the **Boston Housing Dataset**, which includes the following features:
- `CRIM`: Crime rate by town
- `ZN`: Proportion of residential land zoned for large plots
- `INDUS`: Proportion of non-retail business acres per town
- `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- `NOX`: Nitrogen oxides concentration
- `RM`: Average number of rooms per dwelling
- `AGE`: Proportion of owner-occupied units built before 1940
- `DIS`: Weighted distance to employment centers
- `RAD`: Index of accessibility to radial highways
- `TAX`: Property tax rate
- `PTRATIO`: Pupil-teacher ratio
- `B`: Proportion of residents of African American descent
- `LSTAT`: Percentage of lower status population
- `MEDV`: Median value of owner-occupied homes (target variable)

## Steps in the Project
1. **Data Loading**: The dataset is loaded from a CSV file.
2. **Exploratory Data Analysis**:
   - Check for missing values.
   - Analyze basic statistics and correlations between features.
3. **Data Preprocessing**:
   - Split the data into features (`X`) and target variable (`Y`).
   - Split the dataset into training and testing datasets.
   - Standardize the features using `StandardScaler`.
4. **Model Building**:
   - Fit a **Linear Regression** model on the training data.
5. **Model Evaluation**:
   - Evaluate the model using **MAE**, **MSE**, and **RMSE**.
   - Visualize the predicted values against actual values.
6. **Model Coefficients**:
   - Interpret the model coefficients to understand the influence of each feature.

## Prerequisites
To run the project, ensure that the following libraries are installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using `pip`:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
