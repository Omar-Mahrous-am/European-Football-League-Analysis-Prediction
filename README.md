# ⚽ European Football League Predictor

AI-powered football prediction app that forecasts next season's points based on historical performance data.

## Features
- Predict next season points for any major European team
- Uses machine learning model trained on 20+ years of data
- Interactive charts and performance analysis
- No manual input required - just select league and team

## How to Use
1. Select a league from the dropdown
2. Choose your team
3. Click "PREDICT NEXT SEASON POINTS"
4. View the AI-powered prediction with detailed analysis

## Model
- **Algorithm:** Linear Regression
- **Features:** Historical wins, draws, losses, goals for/against
- **Data:** 20+ years of European football data

## 📊 Project Overview

This analysis covers key aspects of football team performance over multiple seasons. The goal is to identify correlations between different variables and use machine learning to predict future rankings.

### 🔍 Main Tasks:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Correlation Matrix & Visualization
- One-Hot Encoding of Categorical Features
- Linear Regression Model
- Model Evaluation Function
- Predictive Insights for Future Seasons

---

## 🧰 Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib & Seaborn**
- **Scikit-learn**

---

## 📁 Dataset

The dataset used in this project contains historical performance data for teams from five major European football leagues.

- File: `5_leages_data.csv`
- Columns include season information, team statistics, league name, and more.

---

## 📈 Example Visuals

- Correlation heatmaps
- Distribution plots for goals, points, and other performance metrics

---

## 🧠 Model

A simple **Linear Regression** model was trained after applying one-hot encoding. A custom function was created to evaluate the model's performance using metrics such as:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

Predictions were made to estimate future season positions for selected teams.

---


