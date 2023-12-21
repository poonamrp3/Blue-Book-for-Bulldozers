import pandas as pd
import matplotlib.pyplot as plt

def cleanData(filename):
    data =  pd.read_csv(filename)
    print(data.info())

    # 1. Remove duplicate values 
    #print(data.shape[0])
    data = removeDuplicates(data)
    '''I can show the number of rows -- to show the after effects of it'''
    # print(data.shape[0])

    # 2. Filter Outliers
    '''For this I can give drop down of the columns where processing can be done. 
    Instead of SalePrice I can detect the possible columns and give the user options '''
    filtered_train_data = filter_outliers(data, 'SalePrice')
    # Check the shape of the datasets to see the impact on the number of rows
    print("\nOriginal Data Shape:", data.shape)
    print("Filtered Data Shape:", filtered_train_data.shape)
    column = 'SalePrice'
    #show_plot(30, column, data, filtered_train_data)

    # 3. Convert date columns
    ''' Dynamic column names'''
    convert_datatype_date(['saledate'], filtered_train_data)
    # create different columns for date
    # Extract useful information from the datetime column, such as year, month, and day
    filtered_train_data = create_date_columns(['saledate'], filtered_train_data)

    # 4. Handle Categorical Data
    # One-hot encode categorical columns
    original_columns = filtered_train_data.columns.tolist()
    processed_data = pd.get_dummies(filtered_train_data, columns=['ProductGroup', 'Enclosure'], drop_first=True)
    #plot_onehotencoded('ProductGroup', filtered_train_data, processed_data, original_columns)
    #plot_onehotencoded('Enclosure', filtered_train_data, processed_data, original_columns)
    
    # 5. Remove Unnecessary columns
    processed_data1 = remove_columns(processed_data)
    return processed_data1

def removeDuplicates(data):
    data.drop_duplicates(inplace=True)
    return data

def filter_outliers(data, column, lower_percentile=0.05, upper_percentile=0.95):
    if column not in data.columns:
        return data 

    # percentile range can also be make dynamic
    lower_bound = data[column].quantile(lower_percentile)
    upper_bound = data[column].quantile(upper_percentile)
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def show_plot(binsize, column, data, filtered_data):
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.title(f'Original data - {column} Distribution')
    data[column].hist(bins=binsize)
    plt.xlabel(column)
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.title(f'Filtered Data - {column} Distribution')
    filtered_data[column].hist(bins=binsize)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def visualize_encoding_effect(original_data, processed_data, column_name):
    plt.figure(figsize=(12, 6))
    
    # Visualize the original data
    plt.subplot(1, 2, 1)
    original_data[column_name].value_counts().plot(kind='bar')
    plt.title(f'Original Data - {column_name}')
    
    # Visualize the processed data
    plt.subplot(1, 2, 2)
    processed_data[column_name].value_counts().plot(kind='bar')
    plt.title(f'Processed Data - {column_name}')
    
    plt.show()

def convert_datatype_date(columns, data):
    for column in columns:
        if data[column].dtype == 'object':
            try:
                data[column] =  pd.to_datetime(data[column])
                print('succcessfully converted')
            except ValueError:
                print('failed to convert')
                pass

def create_date_columns(columns, data):
    for column in columns:
        if data[column].dtype == 'datetime64[ns]':
            datetime_parameters = ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear']
            
            for parameter in datetime_parameters:
                new_column = f"{column.lower()}{parameter}"
                data[new_column] = getattr(data[column].dt, parameter.lower())

        # Drop original column
        data.drop(column, axis=1, inplace=True)  
        return data

def plot_onehotencoded(column, data_before, data_after, original_columns):
    plt.figure(figsize=(12, 6))
    # Plot the original columns
    plt.subplot(1, 2, 1)
    data_before [column].value_counts().plot(kind='bar')
    plt.title(f'Original Data - {column}')
    plt.xlabel('Categories')  # Set x-axis label
    plt.ylabel('Frequency')   # Set y-axis label

    # Plot the one hot encoded columns
    plt.subplot(1, 2, 2)
    data_after[[col for col in data_after.columns if col not in original_columns]].sum().plot(kind = 'bar')
    plt.title(f'One-Hot Encoded Data-{column}')
    plt.xlabel('Categories')  # Set x-axis label
    plt.ylabel('Frequency')   # Set y-axis label
    plt.show()

def remove_columns(data):
    num_columns_before = len(data.columns)
    missing_threshold = 0.3
    missing_percent = data.isnull().mean()
    columns_to_drop = missing_percent[missing_percent > missing_threshold].index
    data.drop(columns_to_drop, axis = 1)
    num_columns_after = len(data.columns)
    print('Columns before remove process: %s' % num_columns_before)
    print('Columns after remove process: %s' % num_columns_after)
    return data
    
def main():
    trainData = cleanData('data/Train.csv')

if __name__ == '__main__':
    main()
    

