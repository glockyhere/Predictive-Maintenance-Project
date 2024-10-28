from src.model_training import train_model
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

if __name__ == "__main__":
    data = pd.read_csv('./data/processed/vehicle_data_featured.csv')
    X = data.drop(columns=['failure_event'])
    y = data['failure_event']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    
    # Save the model
    with open('./src/deploy/model.pkl', 'wb') as f:
        pickle.dump(model, f)
