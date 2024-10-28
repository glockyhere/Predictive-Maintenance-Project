import dash
from dash import dcc, html
import pandas as pd
import pickle
from dash.dependencies import Input, Output

# Load processed data and model
data = pd.read_csv('../data/processed/vehicle_data_featured.csv')
with open('../src/deploy/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Predictive Maintenance Dashboard"),
    
    dcc.Graph(id='sensor-data-graph'),
    
    html.Label("Select Sensor Data to Display"),
    dcc.Dropdown(
        id='sensor-dropdown',
        options=[
            {'label': 'Temperature', 'value': 'temperature'},
            {'label': 'Pressure', 'value': 'pressure'},
            {'label': 'RPM', 'value': 'rpm'},
        ],
        value='temperature'
    ),
    
    html.H2(id='maintenance-alert', style={'color': 'red'}),
    
    dcc.Interval(id='update-interval', interval=1000*60, n_intervals=0)  # Updates every 60 seconds
])

# Callback to update the graph based on sensor data selection
@app.callback(
    Output('sensor-data-graph', 'figure'),
    [Input('sensor-dropdown', 'value')]
)
def update_graph(selected_sensor):
    fig = {
        'data': [
            {'x': data['timestamp'], 'y': data[selected_sensor], 'type': 'line', 'name': selected_sensor}
        ],
        'layout': {
            'title': f"{selected_sensor.capitalize()} Over Time"
        }
    }
    return fig

# Callback to check for predictive maintenance alerts
@app.callback(
    Output('maintenance-alert', 'children'),
    [Input('update-interval', 'n_intervals')]
)
def check_for_maintenance(n):
    # Use the model to make a prediction
    X = data.drop(columns=['failure_event'])
    predictions = model.predict(X)
    
    if predictions[-1] == 1:
        return "Alert: Maintenance Required!"
    else:
        return "Vehicle health is normal."

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
