# Forecasting Smart City Traffic Patterns

**Project for**: Smart City Initiative Internship  
**Author**: [Your Name]  
**Date**: August 26, 2025  

---

## 1. Project Overview
This project aims to analyze and forecast traffic patterns at four key junctions in a city as part of a *Smart City* initiative. By leveraging historical traffic data, we build a predictive model to help the local government manage traffic flow efficiently, anticipate peak hours, and make informed decisions for future infrastructure planning. The model takes into account daily, weekly, and seasonal patterns, as well as the impact of public holidays on traffic volume.

---

## 2. Problem Statement
The government wants to implement a robust traffic management system by understanding and predicting traffic volumes at four major city junctions. The primary goal is to forecast traffic peaks accurately to mitigate congestion and improve citizen services. The solution must account for variations in traffic patterns on normal working days, weekends, and holidays.

---

## 3. Data Description
The dataset consists of two primary files:

- **train_aWnotuB.csv**: Contains hourly traffic data from November 1, 2015, to June 29, 2017, for four junctions.  
  - `DateTime`: The date and hour of the observation.  
  - `Junction`: The junction number (1, 2, 3, or 4).  
  - `Vehicles`: The number of vehicles recorded in that hour (the target variable).  
  - `ID`: A unique identifier for each record.  

- **datasets_84_11879_test_BdBKkAj.csv**: Contains the timestamps and junctions for which we need to forecast traffic, from July 1, 2017, to October 30, 2017.

---

## 4. Methodology

### 4.1. Exploratory Data Analysis (EDA)
We began by loading the data and visualizing it to understand its underlying patterns. Key insights were drawn from plots of traffic over time, hourly distributions, and weekly trends. This step confirmed the presence of strong daily and weekly seasonality and an overall upward trend in traffic volume.

### 4.2. Feature Engineering
To prepare the data for the machine learning model, we extracted several time-based features from the `DateTime` column:
- Year, Month, Day, and Hour  
- Day of the week  
- A binary feature (`Is_Holiday`) to indicate if a given day is a public holiday in India.

### 4.3. Model Building
We used the **LightGBM (Light Gradient Boosting Machine)** algorithm, a powerful and efficient tree-based model well-suited for tabular data and time-series forecasting. The model was trained on the engineered features to predict the `Vehicles` count. A validation set was used to monitor performance and prevent overfitting.

### 4.4. Prediction
The trained model was then used to generate traffic forecasts for the time period specified in the test dataset.

---

## 5. File Structure
```
smart-city-traffic-forecast/
│
├── data/
│   ├── train_aWnotuB.csv
│   └── datasets_84_11879_test_BdBKkAj.csv
│
├── notebooks/
│   └── 1_EDA_and_Visualization.ipynb
│
├── src/
│   └── traffic_forecast.py
│
├── submission.csv
└── README.md
```

---

## 6. Requirements
To run this project, you will need Python 3.x and the following libraries:
- pandas  
- scikit-learn  
- lightgbm  
- holidays  
- matplotlib  
- seaborn  

You can install all the required libraries using pip:
```bash
pip install pandas scikit-learn lightgbm holidays matplotlib seaborn
```

---

## 7. Usage
1. Clone the repository or place all the project files in the same directory.  
2. Ensure you have the data files in a `data/` sub-directory.  
3. Run the main script from your terminal:  
   ```bash
   python src/traffic_forecast.py
   ```  
4. The script will automatically process the data, train the model, and generate a `submission.csv` file in the root directory containing the final traffic predictions.

---

## 8. Results
The model successfully captures the complex patterns in the traffic data and generates forecasts for the specified period. The final output (`submission.csv`) provides the predicted number of vehicles for each hour at each junction, which can be used by the city planners for proactive traffic management and infrastructure planning.

---

## 9. Future Improvements
To further enhance the model's accuracy, the following steps can be considered:
- **Advanced Feature Engineering**: Incorporate lag features (e.g., traffic from the previous hour) and rolling window statistics (e.g., 3-hour moving average).  
- **Hyperparameter Tuning**: Use techniques like Grid Search or Optuna to find the optimal settings for the LightGBM model.  
- **Explore Other Models**: Experiment with time-series specific models like SARIMA or deep learning models like LSTMs.  
- **Incorporate External Data**: Add other relevant data sources, such as weather conditions or information about local events (e.g., festivals, cricket matches), which could influence traffic.  

---
