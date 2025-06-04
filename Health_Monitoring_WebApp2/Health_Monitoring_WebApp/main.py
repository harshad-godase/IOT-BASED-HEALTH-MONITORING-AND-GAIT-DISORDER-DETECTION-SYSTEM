from flask import Flask, request, render_template
import pandas as pd
import requests
import matplotlib.pyplot as plt
import os
import csv
import numpy as np

app = Flask(__name__)

THINGSPEAK_CHANNEL_ID = "2849370"
THINGSPEAK_API_KEY = "3MYAX5Q5ZXG5G5EN"

def fetch_data():
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_API_KEY}&results=10"
    response = requests.get(url)
    return response.json().get("feeds", [])

def load_remedies():
    remedies = {}
    try:
        # Load data from Normal_Data1.csv instead of remedies.csv
        df = pd.read_csv('Gait_disorder.csv')
        
        # Group by Gait_Disorder and get unique Recovery_Suggestion values
        for disorder in df['Gait_Disorder'].unique():
            # Get the first recovery suggestion for each disorder type
            recovery = df[df['Gait_Disorder'] == disorder]['Recovery_Suggestion'].iloc[0]
            remedies[disorder] = recovery
            
        return remedies
    except Exception as e:
        print(f"Error loading remedies: {e}")
        # Return default values if file not found or missing expected columns
        return {
                "Ataxic Gait": "Heel-to-Toe Walking: Walk with one foot in front of the other, touching heel to toe. Duration/Frequency: 10 minutes, 2-3 times a day. Standing Balance: Stand on one leg for 30 seconds, then switch legs. Duration/Frequency: 30 seconds per leg, 3 sets/day.Leg Raises: While sitting, extend one leg straight and hold for a few seconds before lowering. Duration/Frequency: 3 sets of 10 repetitions/leg.",
                "Antalgic Gait": "Gentle Walking: Walk slowly and gently, focusing on smooth movements to reduce pain. Duration/Frequency: 10-15 minutes, 2-3 times/day.Toe Raises: Stand with feet flat on the floor and rise onto your toes to improve strength. Duration/Frequency: 3 sets of 10 reps.Ankle Circles: Sit and rotate your ankles in both clockwise and counterclockwise directions. Duration/Frequency: 3 sets of 10 rotations/leg.",
                "Normal": "Maintain regular exercise and posture training.",
                "Spastic Gait": "Seated Leg Stretch: Sit with one leg extended and gently lean forward to stretch the hamstrings and calves. Duration/Frequency: 3 sets of 20-30 seconds each leg.Standing Quadriceps Stretch: Stand and pull one leg back to stretch the front of the thigh. Duration/Frequency: 3 sets of 20-30 seconds each leg.Hip Flexor Stretch: Lunge forward with one leg to stretch the hip flexors and legs. Duration/Frequency: 3 sets of 20-30 seconds each leg.",
                "Parkinsonian Gait": "Marching in Place: Stand tall and march in place, lifting knees high with each step. Duration/Frequency: 3 sets of 10-15 minutes/day.Heel-to-Toe Walk: Walk in a straight line, placing the heel of one foot directly in front of the toes of the other foot. Duration/Frequency: 10 minutes, 2-3 times per day. Side Stepping: Step sideways in a slow and controlled manner. Duration/Frequency: 3 sets of 10 steps/side, 3 times a day.",
                "Hemiplegic Gait": "Walking with Assistive Devices: Walk using crutches, walkers, or canes to assist with balance and prevent falls. Duration/Frequency: 15-20 minutes/day.Step-ups: Step onto a low step with one leg at a time, alternating legs. Duration/Frequency: 3 sets of 10 steps each leg.Weight Shifting: Shift weight from one leg to the other while standing. Duration/Frequency: 3 sets of 10 shifts/leg.",
        }

def classify_reading(live_x, live_y, live_z):
    try:
        # Load all data from the CSV file
        df = pd.read_csv('Gait_disorder.csv')
        
        # Calculate Euclidean distance for each entry
        df['distance'] = ((df['X_Acceleration'] - live_x)**2 + 
                         (df['Y_Acceleration'] - live_y)**2 + 
                         (df['Z_Acceleration'] - live_z)**2).apply(np.sqrt)
        
        # Find the entry with the minimum distance
        closest_match = df.loc[df['distance'].idxmin()]
        
        # Apply a threshold to determine confidence
        threshold = 0.5  # Adjust based on your data's characteristics
        
        if closest_match['distance'] < threshold:
            return closest_match['Gait_Disorder'], "confident"
        else:
            # If distance is too large, it's an unknown pattern
            return "Abnormal", "uncertain"
            
    except Exception as e:
        print(f"Error in classification: {e}")
        return "Classification Error", "error"

def generate_graphs(x_vals, y_vals, z_vals):
    # Load normal data
    df = pd.read_csv('normal_data3.csv')
    normal_data = df[df['Gait_Disorder'] == 'Normal']
    
    # Find closest normal readings for each live data point
    closest_normal_x = []
    closest_normal_y = []
    closest_normal_z = []
    
    # For each live data point, find the closest normal reading
    for i in range(len(x_vals)):
        live_x, live_y, live_z = x_vals[i], y_vals[i], z_vals[i]
        
        # Calculate distances to all normal readings
        normal_data['distance'] = ((normal_data['X_Acceleration'] - live_x)**2 + 
                                 (normal_data['Y_Acceleration'] - live_y)**2 + 
                                 (normal_data['Z_Acceleration'] - live_z)**2).apply(np.sqrt)
        
        # Find closest normal reading
        if len(normal_data) > 0:
            closest = normal_data.loc[normal_data['distance'].idxmin()]
            closest_normal_x.append(closest['X_Acceleration'])
            closest_normal_y.append(closest['Y_Acceleration'])
            closest_normal_z.append(closest['Z_Acceleration'])
        else:
            # Fallback if no normal data available
            closest_normal_x.append(0)
            closest_normal_y.append(0)
            closest_normal_z.append(9.8)  # Approximate gravity value
    
    # Calculate means and standard deviations for each axis
    mean_x = normal_data['X_Acceleration'].mean()
    mean_y = normal_data['Y_Acceleration'].mean()
    mean_z = normal_data['Z_Acceleration'].mean()
    
    std_x = normal_data['X_Acceleration'].std()
    std_y = normal_data['Y_Acceleration'].std()
    std_z = normal_data['Z_Acceleration'].std()
    
    axes_labels = [
        ("X Accel", x_vals, closest_normal_x, "x_graph.png", mean_x, std_x),
        ("Y Accel", y_vals, closest_normal_y, "y_graph.png", mean_y, std_y),
        ("Z Accel", z_vals, closest_normal_z, "z_graph.png", mean_z, std_z)
    ]
    
    # Generate graphs with closest normal readings
    for label, live_values, normal_values, filename, mean, std in axes_labels:
        plt.figure(figsize=(8, 4))
        plt.plot(normal_values, color='g', label='Normal Pattern')
        plt.plot(live_values, color='r', label='Live Data')
        plt.xlabel("Time")
        plt.ylabel(label)
        plt.legend()
        plt.title(f"{label} Comparison")
        
        # Plot mean ± standard deviation as shaded area
        plt.fill_between(range(len(live_values)), 
                        [mean - std] * len(live_values),
                        [mean + std] * len(live_values),
                        color='g', alpha=0.2)
        
        plt.savefig(f'static/{filename}')
        plt.close()

# Function to determine if readings are normal
def check_readings_status(reading, param):
    if param == 'bpm' and reading>= 50:
        return 'Normal' if 50 <= reading <= 120 else 'Abnormal'
    elif param == 'o2' and reading>= 80:
        return 'Normal' if reading >= 95 else 'Abnormal'
    elif param == 'body_temp' and reading>= 80:
        #return 'Normal' if 36.1 <= reading <= 37.2 else 'Abnormal'
        return 'Normal' if 90 <= reading <= 99 else 'Abnormal'
    return 'Unknown'

@app.route('/', methods=['GET', 'POST'])
def index():
    patient_info = {
        'name': '',
        'age': '',
        'gender': '',
        'weight': ''
    }
    
    latest_data = {}
    status = {}
    
    if request.method == 'POST':
        # Get patient info from form
        patient_info = {
            'name': request.form.get('name', ''),
            'age': request.form.get('age', ''),
            'gender': request.form.get('gender', ''),
            'weight': request.form.get('weight', '')
        }
        
        # Fetch live data from ThingSpeak
        data = fetch_data()
        if data:
            latest_data = data[-1]
            
            # Extract readings from ThingSpeak fields
            # Updated to match the fields shown in the image
            pulse_bpm = float(latest_data.get('field1', 0))
            pulse_ox = float(latest_data.get('field2', 0))
            body_temp = float(latest_data.get('field3', 0))
            
            # Check if readings are normal or abnormal
            status = {
                'bpm': check_readings_status(pulse_bpm, 'bpm'),
                'o2': check_readings_status(pulse_ox, 'o2'),
                'body_temp': check_readings_status(body_temp, 'body_temp')
            }
    
    return render_template("index.html", 
                          patient_info=patient_info,
                          latest_data=latest_data,
                          status=status)

@app.route('/graphs')
def graphs():
    data = fetch_data()
    if not data:
        return "No data available."
    
    # Extract live accelerometer data
    x_vals = [float(d.get('field4', 0)) for d in data]
    y_vals = [float(d.get('field5', 0)) for d in data]
    z_vals = [float(d.get('field6', 0)) for d in data]
    
    generate_graphs(x_vals, y_vals, z_vals)
    
    # Classify the latest reading
    latest_x, latest_y, latest_z = x_vals[-1], y_vals[-1], z_vals[-1]
    classification, status_type = classify_reading(latest_x, latest_y, latest_z)
    
    return render_template(
        'graphs.html',
        x_graph='static/x_graph.png',
        y_graph='static/y_graph.png',
        z_graph='static/z_graph.png',
        classification=classification,  # Single status displayed below graphs
        latest_x=round(latest_x, 3),  # Round to 3 decimal places for display
        latest_y=round(latest_y, 3),
        latest_z=round(latest_z, 3)
    )


@app.route('/disorder_type', methods=['GET', 'POST'])
def disorder_type():
    remedies = load_remedies()
    disorders = list(remedies.keys())
    selected_disorder = ""
    remedy = ""
    
    if request.method == 'POST':
        selected_disorder = request.form.get('disorder', '')
        remedy = remedies.get(selected_disorder, "No remedy found for this disorder.")
    
    return render_template("disorder_type.html", 
                           disorders=disorders,
                           selected_disorder=selected_disorder,
                           remedy=remedy)


if __name__ == '__main__':
    app.run(debug=True)