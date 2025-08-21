import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('sound_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

def detect_sound(sound_data):
    # Preprocess the sound data
    sound_data = scaler.transform(np.array(sound_data).reshape(-1, 1))
    
    # Predict using the trained model
    prediction = model.predict(sound_data)
    
    # Map the prediction to the sound type
    sound_type = {0: 'No Sound', 1: 'Bee Sound', 2: 'Rhinoceros Beetle Sound'}
    return [sound_type[pred] for pred in prediction]

# Example new sound data
new_sound_data = [5, 0.3, 0.2, 0.8, 0.4]  # Replace this with your actual sound data
predictions = detect_sound(new_sound_data)
print(predictions)
