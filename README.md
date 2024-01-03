## README for Machine Learning Project

### Project Overview
This machine learning project focuses on predicting the sale price of heavy machinery based on various features. The key datasets used in this project are `train.csv`. The goal is to develop predictive models that can accurately estimate the sale prices of machines at auction.

### Dataset Description
Link - https://www.kaggle.com/datasets/sureshsubramaniam/bulldozer-training-dataset
#### `train.csv`
- SalesID: Unique identifier for each sale.
- MachineID: Unique identifier for each machine, which can be sold multiple times.
- SalePrice: Sale price of the machine at auction (provided only in `train.csv`).
- saledate: Date of the sale.

#### Additional Information
- There are several fields towards the end of the file related to machine configuration. These options are specific to certain product types, and records for options not applicable to a product type will have null values.

### Data Preprocessing
The `dataprocessing.py` script includes functions for extracting date features, filling missing values in numeric and categorical columns, and preprocessing the data. The script aims to enhance the dataset for model training.

### Model Training and Evaluation
The `main.py` script trains and evaluates machine learning models, including Random Forest, Gradient Boosting, and Linear Regression. The models are saved in the models/ directory, and their performance is assessed through R^2 scores, actual vs. predicted value plots, and feature importance analysis.

### File Structure
- `dataprocessing.py`: Contains functions for data preprocessing.
- `main.py`: Main script for training, evaluating, and saving regression models.
- `data/Train.csv`: Training dataset containing machine sale information.
- `requirements.txt`: Lists the required Python packages for the project.

### How to Run
1. Install dependencies by running `pip install -r requirements.txt`.
2. Run `main.py` to train and evaluate regression models.

Certainly! You can include a summary and a note about which model is better in your README file:

---

### Outut 
Loading model from file: random_forest_model.joblib
Loaded Random Forest Scores:
R^2 Score: 0.9054
Mean Squared Error: 49437415.6250
Mean Absolute Error: 4361.0213

Loading model from file: gradient_boosting_model.joblib
Loaded Gradient Boosting Scores:
R^2 Score: 0.7589
Mean Squared Error: 126049217.9968
Mean Absolute Error: 7626.1023

Loading model from file: linear_regression_model.joblib
Loaded Linear Regression Scores:
R^2 Score: 0.4619
Mean Squared Error: 281299295.5392
Mean Absolute Error: 12027.7451

------

## Model Evaluation and Comparison

### Random Forest Model

- R^2 Score: 0.9054
  - The R^2 score indicates that approximately 90.54% of the variability in SalePrice is explained by the model.
- Mean Squared Error (MSE): 49,437,415.63
  - Lower MSE (49,437,415.63) suggests accurate predictions with minimized errors.
- Mean Absolute Error (MAE): 4,361.02
  - The average absolute difference between predicted and actual SalePrice is $4,361.02.

### Gradient Boosting Model

- R^2 Score: 0.7589
  - The R^2 score indicates that the model accounts for approximately 75.89% of the variability in SalePrice.
- Mean Squared Error (MSE): 126,049,218.00
  - Higher MSE suggests larger errors compared to the Random Forest model.
- Mean Absolute Error (MAE): 7,626.10
  - The average absolute difference between predicted and actual SalePrice is $7,626.10.

### Linear Regression Model

- R^2 Score: 0.4619
  - The model explains approximately 46.19% of the variance in SalePrice.
- Mean Squared Error (MSE): 281,299,295.54
  - Higher MSE indicates larger errors compared to the Random Forest and Gradient Boosting models.
- Mean Absolute Error (MAE): 12,027.75
  - The average absolute difference between predicted and actual SalePrice is $12,027.75.

---
## Model Comparison and Selection

After evaluating the performance of three regression models on the task of predicting machine sale prices, the following observations can be made:

- Random Forest Model:
  - Pros: Achieved the highest R^2 score, indicating better explanatory power. Lower MSE and MAE suggest accurate predictions with minimized errors.
  - Cons: -

- Gradient Boosting Model:
  - Pros: Demonstrated a respectable R^2 score, indicating a good level of explanatory capability. It provides reasonable predictions, albeit with slightly larger errors than the Random Forest model.
  - Cons: Slightly lower performance metrics compared to Random Forest.

- Linear Regression Model:
  - Pros: Achieved a moderate R^2 score, providing some explanatory power.
  - Cons: Relatively lower performance metrics compared to Random Forest and Gradient Boosting.

### Conclusion and Model Selection

Considering the evaluation metrics and overall performance:

- The Random Forest Model outperforms the other models, showing the highest R^2 score and the lowest error metrics. It provides the most accurate and reliable predictions for the given task.

- Therefore, the Random Forest Model is recommended for predicting machine sale prices in this project.

---
### Additional Notes
- Ensure that the correct file paths are specified within the scripts.
- The project assumes a virtual environment for dependency management.

### Author
Poonam Rajan Pawar
