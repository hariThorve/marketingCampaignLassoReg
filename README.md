# Marketing Campaign Effectiveness Prediction

## Project Overview

This project aims to predict the effectiveness of marketing campaigns using machine learning techniques. The primary goal is to identify which campaigns are likely to succeed and estimate the revenue generated from these campaigns. The project utilizes Lasso regression for feature selection and model training.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Key Outputs](#key-outputs)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook (optional for exploration)

## Dataset

The dataset used for this project is `marketing_campaign.xlsx`, which contains demographic and behavioral data related to marketing campaigns. Key features include:

- **Demographic Information**: Age, education, marital status, income, etc.
- **Behavioral Data**: Past purchase history and engagement metrics.
- **Campaign Details**: Type, duration, spend, and outcomes.
- **Target Variable**: Binary indicator of campaign success (`Response`).

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/marketing-campaign-prediction.git
   cd marketing-campaign-prediction
   ```

2. Install the required packages:
   ```bash
   pip install pandas scikit-learn openpyxl matplotlib seaborn
   ```

## Usage

1. Load the dataset and explore the data.
2. Preprocess the data by handling missing values, encoding categorical variables, and scaling numerical features.
3. Define features and target variables.
4. Split the data into training and testing sets.
5. Train the Lasso regression model and perform feature selection.
6. Make predictions and evaluate the model's performance.

### Example Code
