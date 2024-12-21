import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset from Excel
data = pd.read_excel('marketing_campaign.xlsx')

# Step 2: Explore the data
print(data.head())
print(data.info())
print(data.describe())

# Step 3: Handle missing values
# Fill missing values in the 'Income' column with the mean
mean_income = data['Income'].mean()
data['Income'].fillna(mean_income, inplace=True)

# Step 4: Drop the date column
data.drop(columns=['Dt_Customer'], inplace=True)

# Step 5: Feature encoding
data = pd.get_dummies(data, columns=['Education', 'Marital_Status'], drop_first=True)

# Step 6: Feature scaling
scaler = StandardScaler()
numerical_features = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Step 7: Define features and target variable
X = data.drop(['Response'], axis=1)  # Features
y = data['Response']  # Target variable

# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train Lasso regression model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Step 10: Get coefficients and perform feature selection
coefficients = pd.Series(lasso.coef_, index=X.columns)
important_features = coefficients[coefficients != 0].index.tolist()
print("Important Features:", important_features)

# Step 11: Make predictions using the entire test set
y_pred = lasso.predict(X_test)

# Step 12: Evaluate the model
# Convert predictions to binary (0 or 1) for classification
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy}')

# Calculate Mean Squared Error for regression evaluation
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')