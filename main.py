from DataProcessing import *
from model import *

def main():
    train_data_path = 'data/Train.csv'
    processed_train_data = cleanData(train_data_path)

    X_train = processed_train_data.drop('SalePrice', axis=1)
    y_train = processed_train_data['SalePrice']

    knn_classifier(X_train, y_train)
