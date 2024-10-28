import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Load the processed data
    data = pd.read_csv('./data/processed/vehicle_data_featured.csv')
    
    # Split features and target
    X = data.drop(columns=['failure_event'])
    y = data['failure_event']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save the model
    with open('./src/deploy/model.pkl', 'wb') as f:
        pickle.dump(model, f)
