from src.data_preprocessing import load_and_preprocess_data

if __name__ == "__main__":
    data = load_and_preprocess_data('./data/raw/vehicle_data.csv')
    data.to_csv('./data/processed/vehicle_data_processed.csv', index=False)
