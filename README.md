
# Predictive Maintenance for Vehicles

This project implements a **Predictive Maintenance System** for vehicles, utilizing real-time sensor data to predict when maintenance is needed to prevent failures.

## Project Overview

The project focuses on analyzing vehicle sensor data (e.g., **temperature**, **pressure**, **RPM**, **mileage**) to predict **failure events** and alert users when maintenance is due. This helps optimize vehicle performance and reduce unexpected breakdowns.

### Features:
- **Data Collection**: Collects and processes sensor data over time.
- **Feature Engineering**: Includes rolling averages and cumulative mileage for predictive analytics.
- **Machine Learning**: Trains models to predict failures based on historical data.
- **Dashboard**: Visualizes vehicle health in real-time and provides maintenance alerts.
- **API Integration**: Offers a Flask-based API to make predictions via HTTP requests.

## How to Run the Project

### 1. Set up the Environment:
Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

### 2. Run the Project:
- **Preprocess the Data**:
   ```bash
   python scripts/preprocess_data.py
   ```
- **Train the Model**:
   ```bash
   python scripts/train_model.py
   ```
- **Start the Flask API**:
   ```bash
   python deploy/app.py
   ```
- **Run the Dashboard**:
   ```bash
   python dashboard/app.py
   ```

## Directory Structure

```
Predictive_Maintenance_Project/
├── data/                    # Data folder (raw and processed data)
├── deploy/                  # API for serving predictions
├── dashboard/               # Real-time dashboard for monitoring
├── src/                     # Source code (data preprocessing, model training)
├── scripts/                 # Utility scripts for processing and training
└── README.md                # Project overview
```

## Future Improvements
- **Model Optimization**: Improve prediction accuracy with advanced ML models.
- **Scalability**: Integrate with cloud-based services for real-time data ingestion.
- **Expanded Features**: Add support for more vehicle sensors and predictive models.
