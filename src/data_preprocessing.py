import pandas as pd

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    
    # Fill missing values
    data.ffill(inplace=True)
    
    # Normalize data (optional, depending on your data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    sensor_columns = ['temperature', 'pressure', 'rpm']
    data[sensor_columns] = scaler.fit_transform(data[sensor_columns])
    
    return data

if __name__ == "__main__":
    # Example usage
    data = load_and_preprocess_data('./data/raw/vehicle_data.csv')
    data.to_csv('./data/processed/vehicle_data_processed.csv', index=False)
