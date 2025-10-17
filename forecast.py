# =========================================
# AI-Powered Sales Forecasting Dashboard
# =========================================

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# ------------------------------
# Step 0: Setup file paths
# ------------------------------
DATA_FILE = 'retail_sales_dataset.csv'  # Your dataset file
OUTPUT_FOLDER = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'sales_forecast.xlsx')

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset '{DATA_FILE}' loaded successfully!")
except FileNotFoundError:
    print(f"Error: Dataset file '{DATA_FILE}' not found!")
    exit()

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# ------------------------------
# Step 2: Prepare Data for Prophet
# ------------------------------
# Keep only date and total sales, aggregate by day
prophet_df = df.groupby('Date')['Total Amount'].sum().reset_index()

# Rename columns for Prophet
prophet_df = prophet_df.rename(columns={'Date': 'ds', 'Total Amount': 'y'})

print("\nData prepared for Prophet:")
print(prophet_df.head())

# ------------------------------
# Step 3: Train Forecasting Model
# ------------------------------
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(prophet_df)

# Forecast next 90 days
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
print("\nForecast preview (last 5 rows):")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# ------------------------------
# Step 4: Visualize Forecast
# ------------------------------
# Plot forecast
fig1 = model.plot(forecast)
plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()

# Plot forecast components (trend, weekly, yearly)
fig2 = model.plot_components(forecast)
plt.show()

# ------------------------------
# Step 5: Export Forecast for Power BI
# ------------------------------
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel(OUTPUT_FILE, index=False)
print(f"\nForecast exported successfully to '{OUTPUT_FILE}'")
