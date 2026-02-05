#  House Price Prediction using Machine Learning

##  Project Overview

This project implements an **end-to-end machine learning pipeline** to predict house prices using real estate data.  
It covers the complete ML lifecycle — from data understanding and preprocessing to model training, evaluation, explainability, and deployment via a web interface.

The final solution is **production-ready**, interpretable, and scalable.

---

##  Objective

To develop a machine learning model that accurately predicts house prices based on property characteristics such as:

- Location
- Size (sq. ft.)
- Bedrooms and Bathrooms
- Year Built
- Property Condition
- Property Type
- Sale Date

---

##  Dataset Description

The dataset contains the following features:

| Feature     | Description                  |
| ----------- | ---------------------------- |
| Property ID | Unique identifier            |
| Location    | City / geographical location |
| Size        | Area in square feet          |
| Bedrooms    | Number of bedrooms           |
| Bathrooms   | Number of bathrooms          |
| Year Built  | Construction year            |
| Condition   | Property condition at sale   |
| Type        | Property type                |
| Date Sold   | Sale date                    |
| Price       | Sale price (target variable) |

---

##  Approach & Methodology

###  Data Understanding & EDA

- Analyzed price distribution (strong right skew)
- Studied relationships between price and features
- Identified important drivers such as size, location, and age

---

###  Data Cleaning & Preprocessing

- Removed rows with missing target values
- Parsed date fields
- Imputed missing numerical values using **median**
- Applied **label encoding** to categorical features:
  - `Location`
  - `Type`
- Scaled numerical features using **StandardScaler**

---

###  Feature Engineering

- Extracted time-based features:
  - `Sold_Year`
  - `Sold_Month`
  - `Sold_Quarter`
- Created `Property_Age`
- Converted property condition into an ordinal variable
- Applied **log1p transformation** to the target (`Price`) for stability

---

###  Model Training

- Used **5-fold cross-validation**
- Trained and compared a **model zoo**:
  - Ridge, Lasso, ElasticNet
  - Random Forest
  - HistGradientBoosting
  - XGBoost
  - LightGBM
  - CatBoost

---

###  Model Evaluation

Metrics used:

- **MAE (Primary metric)** – most interpretable for business
- RMSE
- R² Score

 **CatBoost** performed best with:

- Lowest MAE
- Stable cross-validation performance
- Strong generalization

---

###  Model Explainability

- Feature importance analysis
- SHAP-based interpretation
- Key price drivers identified:
  - Property size
  - Location
  - Property age
  - Sale year

---

###  Deployment

- Final CatBoost model retrained on **100% of the data**
- Saved artifacts:
  - Trained model
  - Label encoders
  - Imputer
  - Scaler
- Built an interactive **Streamlit web application** for predictions

---

##  Web Application (Streamlit)

The Streamlit app allows users to:

- Enter property details
- Get instant price predictions
- Use the model without technical knowledge

###  Run the App Locally

```bash
# Activate virtual environment
source .venv/bin/activate

# Run Streamlit app
streamlit run app.py
```
