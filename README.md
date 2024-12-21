Marketing Campaign Effectiveness Prediction
Project Overview
This project aims to predict the effectiveness of marketing campaigns using machine learning techniques. The primary goal is to identify which campaigns are likely to succeed and estimate the revenue generated from these campaigns. The project utilizes Lasso regression for feature selection and model training.
Table of Contents
Technologies Used
Dataset
Installation
Usage
Model Evaluation
Key Outputs
Contributing
License
Technologies Used
Python
Pandas
Scikit-learn
NumPy
Matplotlib
Seaborn
Jupyter Notebook (optional for exploration)
Dataset
The dataset used for this project is marketing_campaign.xlsx, which contains demographic and behavioral data related to marketing campaigns. Key features include:
Demographic Information: Age, education, marital status, income, etc.
Behavioral Data: Past purchase history and engagement metrics.
Campaign Details: Type, duration, spend, and outcomes.
Target Variable: Binary indicator of campaign success (Response).
Installation
To set up the project, follow these steps:
1. Clone the repository:
Install the required packages:
Usage
1. Load the dataset and explore the data.
2. Preprocess the data by handling missing values, encoding categorical variables, and scaling numerical features.
Define features and target variables.
4. Split the data into training and testing sets.
5. Train the Lasso regression model and perform feature selection.
Make predictions and evaluate the model's performance.
Example Code
Model Evaluation
The model's performance is evaluated using the following metrics:
Accuracy: Measures the proportion of correct predictions.
Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.
Key Outputs
Important Features: Identified features that significantly impact campaign success.
Accuracy: The model achieved an accuracy of 84.60%.
Mean Squared Error: The model's MSE was 0.1261.
