import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from DataProcessing import preprocess_data
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def save_and_load_model(model, x_val, y_val, model_name="Model", color='blue', sample_size=None):
    """
    Show scores, plot actual vs. predicted values, and feature importance for a loaded model.

    Parameters:
    - model: Loaded regression model.
    - x_val: Validation set features.
    - y_val: Validation set labels.
    - model_name: Name of the model.
    - color: Color for plotting.
    - sample_size: Number of samples to display in the plot.
    """
    # Show scores for the loaded model
    show_scores(model, x_val, y_val, f"Loaded {model_name}")

    # Plot actual vs. predicted values for the loaded model
    plot_actual_vs_predicted(model, x_val, y_val, f"Loaded {model_name}", color=color, sample_size=sample_size)

    # Plot feature importance for the loaded model
    plot_feature_importance(model, x_val.columns, f"Loaded {model_name}")

def show_scores(model, x_val, y_val, model_name="Model"):
    """
    Display R^2 score for the loaded model.

    Parameters:
    - model: Loaded regression model.
    - x_val: Validation set features.
    - y_val: Validation set labels.
    - model_name: Name of the model.
    """
    val_preds = model.predict(x_val)
    '''scores = {"Valid R^2": r2_score(y_val, val_preds)}
    print(f"{model_name} Scores:")
    for score_name, score_value in scores.items():
        print(f"{score_name}: {score_value:.4f}")'''
    r2 = r2_score(y_val, val_preds)
    mse = mean_squared_error(y_val, val_preds)
    mae = mean_absolute_error(y_val, val_preds)

    # Display the metrics
    print(f"{model_name} Scores:")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

def plot_actual_vs_predicted(model, x_val, y_val, model_name="Model", color='blue', sample_size=None):
    """
    Plot actual vs. predicted values for the loaded model.

    Parameters:
    - model: Loaded regression model.
    - x_val: Validation set features.
    - y_val: Validation set labels.
    - model_name: Name of the model.
    - color: Color for plotting.
    - sample_size: Number of samples to display in the plot.
    """
    val_preds = model.predict(x_val)

    if sample_size:
        y_val = y_val[:sample_size]
        val_preds = val_preds[:sample_size]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_val, y=val_preds, label=model_name, color=color)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name} - Actual vs. Predicted Values")
    plt.legend()
    plt.show()

def plot_feature_importance(model, feature_names, model_name="Model", top_n=10):
    """
    Plot top feature importance for the loaded model.

    Parameters:
    - model: Loaded regression model.
    - feature_names: Names of features.
    - model_name: Name of the model.
    - top_n: Number of top features to display in the plot.
    """
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(top_n)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(f"{model_name} - Top {top_n} Feature Importance")
        plt.show()
        
def load_or_train_model(model_filename, model_instance, x_train, y_train):
    """
    Load a pre-trained model or train a new model and save it.

    Parameters:
    - model_filename: File name for saving or loading the model.
    - model_instance: Instance of the regression model.
    - x_train: Training set features.
    - y_train: Training set labels.

    Returns:
    Loaded or newly trained regression model.
    """
    if os.path.exists(model_filename):
        print(f"Loading model from file: {model_filename}")
        loaded_model = joblib.load(model_filename)
    else:
        print(f"Training model and saving to {model_filename}")
        loaded_model = model_instance
        loaded_model.fit(x_train, y_train)
        joblib.dump(loaded_model, model_filename)
        print(f"Model trained and saved to {model_filename}")
    return loaded_model

def main():
    """
    Main function to train, evaluate, and plot different regression models.
    """
    train_data_path = 'data/Train.csv'

    # Read and preprocess the data
    df = pd.read_csv(train_data_path, low_memory=False, parse_dates=["saledate"])
    df = preprocess_data(df)

    # Display missing values information
    df.isna().sum()

    # Split the data into training and validation sets
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    # Split data into x & y
    x_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
    x_val, y_val = df_val.drop("SalePrice", axis=1), df_val.SalePrice

    # Random Forest Model
    rf_model_name = "Random Forest"
    rf_model_filename = f'{rf_model_name.lower().replace(" ", "_")}_model.joblib'

    # Load or train the Random Forest Model
    loaded_rf_model = load_or_train_model(rf_model_filename, RandomForestRegressor(n_jobs=-1, random_state=42),
                        x_train, y_train)

    # Evaluate and plot Random Forest Model
    save_and_load_model(loaded_rf_model, x_val, y_val, rf_model_name, color='blue', sample_size=1000)

    # Gradient Boosting Model
    gb_model_name="Gradient Boosting"
    gb_model_filename = f'{gb_model_name.lower().replace(" ","_")}_model.joblib'

    # Load or train the Gradient Boosting Model
    loaded_gb_model = load_or_train_model(gb_model_filename,GradientBoostingRegressor(random_state=42),
                       x_train, y_train)

    # Evaluate and plot Gradient Boosting Model
    save_and_load_model(loaded_gb_model, x_val, y_val, "Gradient Boosting", color='green', sample_size=1000)     

    # Linear Regression Model
    lr_model_name = "Linear Regression"
    lr_model_filename = f'{lr_model_name.lower().replace(" ","_")}_model.joblib'
    loaded_lr_model = load_or_train_model(lr_model_filename, LinearRegression(), x_train, y_train)

    # Evaluate and plot Linear Regression Model
    save_and_load_model(loaded_lr_model, x_val, y_val, "Linear Regression", color='red', sample_size=1000)

if __name__ == '__main__':
    main()