from src.model_evaluation import evaluate_model
import pandas as pd
import pickle

if __name__ == "__main__":
    data = pd.read_csv('./data/processed/vehicle_data_featured.csv')
    X = data.drop(columns=['failure_event'])
    y = data['failure_event']

    with open('./src/deploy/model.pkl', 'rb') as f:
        model = pickle.load(f)

    evaluate_model(X, y, model)
