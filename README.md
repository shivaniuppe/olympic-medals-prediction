# Olympic Medals Prediction

This project predicts the number of medals a country will win in the Olympics based on historical data. It uses machine learning techniques to analyze features such as the number of athletes, previous medals, and age of participants to make predictions.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Setup Instructions](#setup-instructions)
4. [Running the Project](#running-the-project)
5. [Results](#results)
6. [Dependencies](#dependencies)

---

## Project Overview

The goal of this project is to predict the number of medals a country will win in the Olympics using historical data. The project follows a standard machine learning workflow:
1. **Data Preparation**: The raw athlete-level data is processed to create a team-level dataset.
2. **Exploratory Data Analysis (EDA)**: The dataset is explored to understand relationships between features and the target variable.
3. **Model Training**: A linear regression model is trained to predict the number of medals.
4. **Model Evaluation**: The model's performance is evaluated using mean absolute error (MAE).

---

## Dataset

The dataset used in this project is from the historical Olympic games. It includes two files:
1. **`athlete_events.csv`**: Original athlete-level data containing information about individual athletes, their events, and medals won.
2. **`teams.csv`**: Team-level data generated from `athlete_events.csv`. It contains aggregated features like `team`, `country`, `year`, `athletes`, `age`, `prev_medals`, and `medals`.

The dataset is available on [Kaggle](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results).

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/olympic-medals-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd olympic-medals-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Project

The project is implemented in a single Jupyter notebook: **`olympic_medals_prediction.ipynb`**. The notebook is divided into two main sections:

### 1. Data Preparation
- Load the raw athlete-level data (`athlete_events.csv`).
- Aggregate the data to create a team-level dataset (`teams.csv`).
- Handle missing values and clean the data.

### 2. Machine Learning Model
- Perform exploratory data analysis (EDA) to understand the dataset.
- Split the data into training and testing sets.
- Train a linear regression model to predict the number of medals.
- Evaluate the model's performance using mean absolute error (MAE).

To run the notebook:
1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook notebooks/olympic_medals_prediction.ipynb
   ```
2. Run the cells sequentially to execute the data preparation and machine learning steps.

---

## Results

The model predicts the number of medals using features like `athletes` and `prev_medals`. The performance is evaluated using **Mean Absolute Error (MAE)**. Key findings include:
- The model achieves an MAE of `X` (replace with your actual error value).
- The error ratio varies by team, with some teams having more accurate predictions than others.

### Example Predictions
- **USA**: Predicted `X` medals, actual `Y` medals.
- **IND**: Predicted `X` medals, actual `Y` medals.

For more details, refer to the `olympic_medals_prediction.ipynb` notebook.

---

## Dependencies

The project requires the following Python packages:
- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `jupyter`

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Acknowledgments
- Dataset sourced from [Kaggle](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results).
