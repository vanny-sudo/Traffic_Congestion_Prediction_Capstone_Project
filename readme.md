# 🚦 Traffic Congestion Prediction – Capstone Project

---

## 👤 **Personal Information**

**Names:** Irakoze Grace Vanny

**ID:** 26425

**Course:** Introduction to Big Data Analytics

**Email:** vannygrace2020@gmail.com

---

## 📌 **Project Overview**

This project predicts **traffic congestion patterns and peak hours** using **Big Data Analytics** in the **Transportation sector**.
By analyzing traffic data, the project aims to:

- 🕒 Identify **peak traffic hours and busiest days**
- 🚗 Help **drivers inorder to reduce travel delays**
- 🌱 Support **city planners in traffic management & reducing emissions**

**Dataset:** [Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)

---

## 1️⃣ **Part 1: Problem Definition & Planning**

### ✅ **Sector Selection**

☑ **Transportation**

### ✅ **Problem Statement**

Traffic congestion is a **daily challenge** in many cities, especially during **morning and evening rush hours**, resulting in:

- ⏱ **Delays in travel**
- ⛽ **Wasted fuel**
- 🌍 **Air pollution**

**Objective:**

> *"Predict traffic jam patterns and peak hours using transportation data to reduce city travel delays."*

### ✅ **Dataset Identification**

- **Title:** Metro Interstate Traffic Volume
- **Rows × Columns:** \~48,000 × 9
- **Data Type:** Structured (CSV)
- **Status:** Requires Preprocessing

### ✅ **Planning Steps**

1. Collect traffic volume data from UCI Repository.
2. Clean and preprocess data (handle missing values, duplicates, convert timestamps).
3. Analyze traffic patterns to find **peak hours and days**.
4. Build prediction model using **Random Forest Regression**.
5. Create **Power BI dashboard** with interactive visualizations.

---

## 2️⃣ **Part 2: Python Analytics Tasks**

### 🧹 **Data Cleaning**

- Handle **missing values** & **duplicates**
- Convert **date\_time** to datetime
- Extract **hour** and **weekday** features
- Remove **outliers**

```python
import pandas as pd

# Load dataset
data = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# Clean dataset
data.dropna(inplace=True)
data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].dt.hour
data['weekday'] = data['date_time'].dt.weekday
data.drop_duplicates(inplace=True)
```

### 📊 **Exploratory Data Analysis (EDA)**

- Visualize **traffic distribution**
- Compare **traffic by hour and weekday**
- Create **heatmaps** for peak traffic

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data['traffic_volume'], bins=30, kde=True)
plt.title("Traffic Volume Distribution")
plt.show()
```

### 🤖 **Machine Learning Model**

- **Algorithm:** Random Forest Regression
- **Features:** hour, weekday, temp, rain\_1h, snow\_1h, clouds\_all
- **Target:** traffic\_volume

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = data[['hour','weekday','temp','rain_1h','snow_1h','clouds_all']]
y = data['traffic_volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 📈 **Model Evaluation**

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R² Score:", r2)
```

### 💡 **Innovations**

- Suggest **best travel hour** for each day

```python
def best_travel_hour(weekday):
    subset = data[data['weekday'] == weekday]
    avg_by_hour = subset.groupby('hour')['traffic_volume'].mean()
    return avg_by_hour.idxmin()

print("Best travel hour on Monday:", best_travel_hour(0))
```

---

## 3️⃣ **Part 3: Power BI Dashboard Tasks**

### 🎯 **Dashboard Requirements**

1. **Communicate insights clearly** with title and problem summary.
2. **Interactive slicers and filters** for weekday, hour, and weather.
3. **Visualizations:**
   - 📊 Bar chart: Traffic by weekday
   - 📈 Line chart: Traffic by hour
   - 🌡 Heatmap: Average traffic per weekday × hour
   - 📋 KPI cards: Peak volume, Avg volume
4. **Innovative Features:**
   - DAX formulas for **dynamic KPIs**
   - Bookmarks for **peak vs off-peak** view
   - Optional: AI Insights visual
   
  
   <img width="959" height="503" alt="image" src="https://github.com/user-attachments/assets/51d9eeac-9d78-4625-a585-bb523e28e4e7" />


---

## 5️⃣ **Complexity & Creativity (Optional)**

- Combine **traffic + weather** for enhanced predictions.
- Apply **Time Series Forecasting (ARIMA/Prophet)**.
- Add **custom DAX measures** for live KPIs.

---

## 6️⃣ **Contact Infos**


- Contact: **vannygrace2020@gmail.com**

---
