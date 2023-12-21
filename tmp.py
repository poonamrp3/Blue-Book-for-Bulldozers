import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the training data
train_data = pd.read_csv('data/Train.csv')

# 1. Remove Duplicate Observations
train_data.drop_duplicates(inplace=True)

# 2. Filter Unwanted Outliers
# Example: Remove outliers in the 'SalePrice' column
lower_bound = train_data['SalePrice'].quantile(0.05)
upper_bound = train_data['SalePrice'].quantile(0.95)
train_data = train_data[(train_data['SalePrice'] >= lower_bound) & (train_data['SalePrice'] <= upper_bound)]

# 3. Fix Structural Errors
# Example: Standardize values in the 'UsageBand' column
train_data['UsageBand'] = train_data['UsageBand'].replace({'N/A': 'Not Applicable'})

# 4. Fix Missing Data
# Example: Impute missing values in the 'auctioneerID' column with mean
train_data['auctioneerID'].fillna(train_data['auctioneerID'].mean(), inplace=True)

# 5. Validate Your Data
# Answer key questions to validate the quality of your prepped data.

# 6. Convert Date Columns
# Convert the 'saledate' column to datetime format for better handling
train_data['saledate'] = pd.to_datetime(train_data['saledate'])

# 7. Feature Engineering
# Extract useful information from the datetime column, such as year, month, and day
train_data['sale_year'] = train_data['saledate'].dt.year

# 8. Handle Categorical Data
# One-hot encode categorical columns
train_data = pd.get_dummies(train_data, columns=['ProductGroup', 'Enclosure', ...], drop_first=True)

# 9. Drop Unnecessary Columns
# Drop columns with a large number of missing values or those not needed for modeling
columns_to_drop = ['fiModelSeries', 'fiModelDescriptor', ...]
train_data = train_data.drop(columns_to_drop, axis=1)

# 10. Handle Outliers (Optional)
# Handle outliers in numerical columns if necessary

# 11. Scaling (Optional, if needed)
# Scale numerical features if required by the model
scaler = StandardScaler()
numerical_columns = ['YearMade', 'MachineHoursCurrentMeter']
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])

# 12. Save Cleaned Data
# Save cleaned data to a new CSV file
train_data.to_csv('cleaned_train_data.csv', index=False)
