import pandas as pd

def feature_engineering(data):
    # Time since last maintenance
    data['time_since_last_maintenance'] = data['timestamp'].diff().fillna(0)

    # Rolling averages for sensor data
    data['avg_temp_7d'] = data['temperature'].rolling(window=7).mean().fillna(data['temperature'].mean())

    # Cumulative mileage
    data['cumulative_mileage'] = data['mileage'].cumsum()

    return data

if __name__ == "__main__":
    # Load processed data
    data = pd.read_csv('./data/processed/vehicle_data_processed.csv')

    # Perform feature engineering
    data = feature_engineering(data)

    # Save the processed data
    data.to_csv('./data/processed/vehicle_data_featured.csv', index=False)
