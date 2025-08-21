from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preprocessing import load_and_preprocess_data

# File path to the dataset
file_path = 'dataset.csv'

# Load and preprocess the data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model and scaler to disk
joblib.dump(model, 'sound_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
