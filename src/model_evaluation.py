import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(X_test, y_test, model):
    # Predict
    y_pred = model.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.show()

if __name__ == "__main__":
    # Load test data
    data = pd.read_csv('./data/processed/vehicle_data_featured.csv')

    # Split data
    X = data.drop(columns=['failure_event'])
    y = data['failure_event']
    
    # Load model
    with open('./src/deploy/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Evaluate the model
    evaluate_model(X, y, model)
