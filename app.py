# main.py
import os
import pickle
import smtplib
from email.mime.text import MIMEText
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import numpy as np
import re
import pandas as pd
import json
import random
from flask_mail import Mail, Message # Import Message from Flask-Mail
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import os
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler
from backend import initialize_llm, load_or_create_vector_db, setup_qa_chain
from itsdangerous import URLSafeTimedSerializer as Serializer
import forms 

app = Flask(__name__)

# Get absolute path to the DB file
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'database.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SECRET_KEY'] = 'thisisme'


# Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'sannidhiyk1@gmail.com'
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS')
mail = Mail(app)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

from models import User, Mood, Medication, QueryHistory, JournalEntry

# Initialize chatbot components when the Flask app starts
llm = initialize_llm()
vector_db = load_or_create_vector_db()
qa_chain = setup_qa_chain(vector_db, llm)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

import routes

def get_reset_token(self, expires_sec=1800):
    s = Serializer(app.config['SECRET_KEY'])
    return s.dumps({'user_id': self.id})
    
@staticmethod
def verify_reset_token(token, expires_sec=1800):
    s = Serializer(app.config['SECRET_KEY'])
    try:
        user_id = s.loads(token, max_age=expires_sec)['user_id']
    except Exception:
        return None
    return User.query.get(user_id)

with app.app_context():
    with db.session.begin_nested():
        db.create_all()
        print("✅ Table created successfully")
        print("Database path →", db_path)
    db.create_all()
    print("✅ Table created successfully")
    print("Database path →", db_path)

    # --- Helper Function for Text Normalization ---
def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)  # Convert to string if not already
    text = text.lower()
    text = text.replace('\xa0', ' ')  # Replace non-breaking space with regular space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    return text

# ... other parts of your main.py ...

# Load data
sym_des = pd.read_csv('datasets/symtoms_df.csv')
precautions = pd.read_csv('datasets/precautions_df.csv')
workout = pd.read_csv('datasets/workout_df.csv')
description = pd.read_csv('datasets/description.csv')
medication_df = pd.read_csv('datasets/medications.csv')
diet = pd.read_csv('datasets/diets.csv')
training_data = pd.read_csv("datasets/Training.csv")  # Load training data

# Separate features and target variable from training data
X_train = training_data.drop(columns=['prognosis'])
y_train = training_data['prognosis']

# Encode categorical labels (target variable)
label_encoder_train = LabelEncoder()
y_train_encoded = label_encoder_train.fit_transform(y_train)

# Initialize and fit the StandardScaler on the training features ONLY
scaler_train = pickle.load(open("models/scaler.pkl", 'rb'))  # Load the scaler
X_train_scaled = scaler_train.transform(X_train)

# Load trained model and label encoder
loaded_model = pickle.load(open("models/best_model.pkl", 'rb'))
loaded_label_encoder = pickle.load(open("models/label_encoder.pkl", 'rb'))

all_symptoms_list = list(pd.read_csv("datasets/Training.csv").columns[:-1])

symptoms_dict_raw = {'itching': 0, 'skin rash': 1, 'nodal skin eruptions': 2, 'continuous sneezing': 3, 'shivering': 4, 'chills': 5,
                    'joint pain': 6, 'stomach pain': 7, 'acidity': 8, 'ulcers on tongue': 9, 'muscle wasting': 10, 'vomiting': 11,
                    'burning micturition': 12, 'spotting urination': 13, 'fatigue': 14, 'weight gain': 15, 'anxiety': 16,
                    'cold hands and feets': 17, 'mood swings': 18, 'weight loss': 19, 'restlessness': 20, 'lethargy': 21,
                    'patches in throat': 22, 'irregular sugar level': 23, 'cough': 24, 'high fever': 25, 'sunken eyes': 26,
                    'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish skin': 32,
                    'dark urine': 33, 'nausea': 34, 'loss of appetite': 35, 'pain behind the eyes': 36, 'back pain': 37,
                    'constipation': 38, 'abdominal pain': 39, 'diarrhoea': 40, 'mild fever': 41, 'yellow urine': 42,
                    'yellowing of eyes': 43, 'acute liver failure': 44, 'fluid overload': 45, 'swelling of stomach': 46,
                    'swelled lymph nodes': 47, 'malaise': 48, 'blurred and distorted vision': 49, 'phlegm': 50,
                    'throat irritation': 51, 'redness of eyes': 52, 'sinus pressure': 53, 'runny nose': 54, 'congestion': 55,
                    'chest pain': 56, 'weakness in limbs': 57, 'fast heart rate': 58, 'pain during bowel movements': 59,
                    'pain in anal region': 60, 'bloody stool': 61, 'irritation in anus': 62, 'neck pain': 63, 'dizziness': 64,
                    'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen legs': 68, 'swollen blood vessels': 69,
                    'puffy face and eyes': 70, 'enlarged thyroid': 71, 'brittle nails': 72, 'swollen extremeties': 73,
                    'excessive hunger': 74, 'extra marital contacts': 75, 'drying and tingling lips': 76, 'slurred speech': 77,
                    'knee pain': 78, 'hip joint pain': 79, 'muscle weakness': 80, 'stiff neck': 81, 'swelling joints': 82,
                    'movement stiffness': 83, 'spinning movements': 84, 'loss of balance': 85, 'unsteadiness': 86,
                    'weakness of one body side': 87, 'loss of smell': 88, 'bladder discomfort': 89, 'foul smell of urine': 90,
                    'continuous feel of urine': 91, 'passage of gases': 92, 'internal itching': 93, 'toxic look (typhos)': 94,
                    'depression': 95, 'irritability': 96, 'muscle pain': 97, 'altered sensorium': 98, 'red spots over body': 99,
                    'belly pain': 100, 'abnormal menstruation': 101, 'dischromic patches': 102, 'watering from eyes': 103,
                    'increased appetite': 104, 'polyuria': 105, 'family history': 106, 'mucoid sputum': 107, 'rusty sputum': 108,
                    'lack of concentration': 109, 'visual disturbances': 110, 'receiving blood transfusion': 111,
                    'receiving unsterile injections': 112, 'coma': 113, 'stomach bleeding': 114, 'distention of abdomen': 115,
                    'history of alcohol consumption': 116, 'fluid overload 1': 117, 'blood in sputum': 118,
                    'prominent veins on calf': 119, 'palpitations': 120, 'painful walking': 121, 'pus filled pimples': 122,
                    'blackheads': 123, 'scurring': 124, 'skin peeling': 125, 'silver like dusting': 126,
                    'small dents in nails': 127, 'inflammatory nails': 128, 'blister': 129, 'red sore around nose': 130,
                    'yellow crust ooze': 131}
symptoms_dict = {normalize_text(k): v for k, v in symptoms_dict_raw.items()}
num_total_features = len(training_data.columns) - 1


# --- Helper Function to Get Disease Information ---
def helper(dis_name_normalized):
    # dis_name_normalized should already be normalized from get_predicted_value or main loop
    print(f"Looking up disease: '{dis_name_normalized}'")

    # Normalize 'Disease' columns in each DataFrame for robust lookup
    description['Disease'] = description['Disease'].astype(str).apply(normalize_text)
    precautions['Disease'] = precautions['Disease'].astype(str).apply(normalize_text)
    medication_df['Disease'] = medication_df['Disease'].astype(str).apply(normalize_text)
    diet['Disease'] = diet['Disease'].astype(str).apply(normalize_text)

    workout_col_to_use = None
    # Check for 'Disease' or 'disease' column in workout DataFrame
    workout_columns_lower = [col.lower() for col in workout.columns]
    if 'disease' in workout_columns_lower:  # Prioritize 'disease' if 'Disease' is also there but different case
        original_col_name = workout.columns[workout_columns_lower.index('disease')]
        workout_col_to_use = original_col_name
        workout[workout_col_to_use] = workout[workout_col_to_use].astype(str).apply(normalize_text)
    elif 'Disease' in workout.columns:  # Check original case if 'disease' not found
        workout_col_to_use = 'Disease'
        workout[workout_col_to_use] = workout[workout_col_to_use].astype(str).apply(normalize_text)
    else:
        print("Warning: Workout DataFrame does not have a 'Disease' or 'disease' column.")

    # Description lookup
    desc_df = description[description['Disease'] == dis_name_normalized]
    desc = desc_df['Description'].iloc[0] if not desc_df.empty else "No description available."

    # Precautions lookup
    pre_df = precautions[precautions['Disease'] == dis_name_normalized]
    if not pre_df.empty:
        # Fill NaN with a placeholder string before converting to list
        pre_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
        pre = pre_df[pre_cols].fillna("N/A").values.tolist()  # Ensure all 4 are present, fill NaN
    else:
        pre = [["No precautions available.", "N/A", "N/A", "N/A"]]  # Consistent list of lists

    # Medication lookup
    med_df = medication_df[medication_df['Disease'] == dis_name_normalized]
    # Assuming 'Medication' column contains lists as strings like "['med1', 'med2']"
    # If it's just a string, ast.literal_eval might be needed, but let's assume it's handled or just a string
    med = med_df['Medication'].fillna("No medication information available.").values.tolist() if not med_df.empty else ["No medication information available."]

    # Diet lookup
    diet_df = diet[diet['Disease'] == dis_name_normalized]
    # Assuming 'Diet' column contains lists as strings
    die = diet_df['Diet'].fillna("No dietary recommendations available.").values.tolist() if not diet_df.empty else ["No dietary recommendations available."]

    # Workout recommendations lookup
    wrkout = ["No workout recommendations available."]  # Default
    if workout_col_to_use:
        wrkout_df = workout[workout[workout_col_to_use] == dis_name_normalized]
        if not wrkout_df.empty:
            # Assuming 'workout' column contains individual workout strings or lists as strings
            wrkout = wrkout_df['workout'].fillna("N/A").values.tolist()
    else:
        wrkout = ["Workout data configuration issue (column not found)."]

    return desc, pre, med, die, wrkout

# --- Prediction Function ---
def get_predicted_value(patient_symptoms, trained_model, trained_scaler, trained_label_encoder, symptoms_feature_map, num_total_features_expected):
    input_vector = np.zeros(num_total_features_expected)
    processed_symptoms = []

    for item_raw in patient_symptoms:
        item_normalized = normalize_text(item_raw)
        if item_normalized in symptoms_feature_map:
            feature_index = symptoms_feature_map[item_normalized]
            # Ensure the index from symptoms_dict is valid for the input_vector
            if 0 <= feature_index < num_total_features_expected:
                input_vector[feature_index] = 1
                processed_symptoms.append(item_normalized)  # Store normalized symptom
            else:
                print(f"Warning: Symptom '{item_normalized}' maps to an out-of-bounds index {feature_index}. Max index is {num_total_features_expected - 1}.")
        else:
            print(f"Warning: Symptom '{item_normalized}' (from input '{item_raw}') not found in the known symptom list.")

    if not processed_symptoms and patient_symptoms:
        print("Warning: None of the provided symptoms were recognized. Prediction might be unreliable.")
    elif not patient_symptoms:
        print("Warning: No symptoms provided. Prediction might be unreliable.")

    input_vector_reshaped = input_vector.reshape(1, -1)
    # The UserWarning "X does not have valid feature names..." occurs because
    # trained_scaler was fit on a DataFrame (X_train, with feature names)
    # and here we pass a NumPy array. This is generally fine as scaling is positional.
    input_vector_scaled = trained_scaler.transform(input_vector_reshaped)

    predicted_disease_encoded = trained_model.predict(input_vector_scaled)
    # Normalize the output of the label encoder
    predicted_disease_raw = trained_label_encoder.inverse_transform(predicted_disease_encoded)[0]
    predicted_disease_normalized = normalize_text(predicted_disease_raw)

    return predicted_disease_normalized, processed_symptoms


def infer_mood_critically(answers):
    happy = answers['happy']
    sad = answers['sad']
    anxious = answers['anxious']
    angry = answers['angry']
    calm = answers['calm']
    energetic = answers['energetic']
    tired = answers['tired']
    focused = answers['focused']
    stress = answers['stress_level']
    comfort = answers['physical_comfort'] # Correctly assigned to 'comfort'
    social = answers['social_interaction']

    mood_scores = {
        "Happy": happy,
        "Sad": sad,
        "Anxious": anxious,
        "Angry": angry,
        "Calm": calm,
        "Energetic": energetic,
        "Tired": tired,
        "Focused": focused
    }

    primary_mood = max(mood_scores, key=mood_scores.get, default="Neutral")
    inferred_mood = primary_mood

    # Refine based on intensity and combinations (more sophisticated logic needed)
    if happy > 3 and sad < 2 and anxious < 2 and angry < 2:
        if energetic > tired:
            inferred_mood = "Happy and Energetic"
        elif calm > anxious:
            inferred_mood = "Happy and Calm"
        else:
            inferred_mood = "Happy"
    elif sad > 3 and happy < 2:
        if tired > energetic:
            inferred_mood = "Sad and Tired"
        else:
            inferred_mood = "Sad"
    elif anxious > 3 and calm < 2:
        if stress in ["High", "Very High"]:
            inferred_mood = "Highly Anxious"
        else:
            inferred_mood = "Anxious"
    elif angry > 3 and calm < 2:
        if stress in ["High", "Very High"]:
            inferred_mood = "Highly Irritable"
        else:
            inferred_mood = "Irritable"
    elif calm > 3 and anxious < 2 and angry < 2:
        if focused > 3:
            inferred_mood = "Calm and Focused"
        else:
            inferred_mood = "Calm"
    elif tired > 3 and energetic < 2:
        inferred_mood = "Low Energy"

    # Consider stress, comfort, social interaction (basic example)
    if stress in ["High", "Very High"] and "Anxious" not in inferred_mood and "Irritable" not in inferred_mood:
        inferred_mood = f"Stressed and {inferred_mood}"
    elif comfort in ["Uncomfortable", "Very Uncomfortable"] and "Sad" not in inferred_mood and "Irritable" not in inferred_mood: # Using the correct variable 'comfort'
        inferred_mood = f"Physically Uncomfortable and {inferred_mood}"
    elif social in ["Negative", "Very Negative"] and "Sad" not in inferred_mood and "Angry" not in inferred_mood and "Anxious" not in inferred_mood:
        inferred_mood = f"Socially Affected and {inferred_mood}"

    return inferred_mood

def get_mood_suggestion(mood, answers=None):
    suggestions = {
        'Happy': ["Enjoy your positive feelings.", "Consider sharing your happiness."],
        'Happy and Energetic': ["Make the most of your energy!", "Engage in an active hobby."],
        'Happy and Calm': ["Savor this peaceful joy.", "Continue with relaxing activities."],
        'Sad': ["Allow yourself to feel your emotions.", "Reach out to a supportive person."],
        'Sad and Tired': ["Be gentle with yourself and rest.", "Consider a comforting activity."],
        'Anxious': ["Try deep breathing exercises.", "Listen to calming music."],
        'Highly Anxious': ["Focus on grounding techniques.", "Consider a guided meditation."],
        'Irritable': ["Take a break to cool down.", "Try a calming breathing exercise."],
        'Highly Irritable': ["Find a healthy way to release your frustration.", "Avoid triggers if possible."],
        'Calm': ["Savor this peaceful moment.", "Engage in a relaxing activity."],
        'Calm and Focused': ["Use this clarity to focus on tasks.", "Enjoy your mental sharpness."],
        'Low Energy': ["Try a light walk or stretch.", "Make sure you're hydrated."],
        'Stressed and Happy': ["Try to maintain your positive mood despite stress.", "Practice stress-reducing activities."],
        'Stressed and Sad': ["Be kind to yourself during this difficult time.", "Consider talking to someone."],
        'Stressed and Anxious': ["Prioritize stress-reduction techniques.", "Limit stressors if possible."],
        'Physically Uncomfortable and Happy': ["Try to find comfort while enjoying your mood.", "Engage in gentle activities."],
        'Physically Uncomfortable and Sad': ["Focus on comfort and self-care.", "Rest and allow yourself to heal."],
        'Socially Affected and Happy': ["Consider connecting with supportive people.", "Share your positive feelings."],
        'Socially Affected and Sad': ["Reach out for connection and support.", "Limit negative social interactions."],
        'Neutral': ["Observe your feelings without judgment.", "Engage in a light activity."],
        'Other': ["Reflect on this feeling.", "Consider journaling about it."],
    }
    return random.choice(suggestions.get(mood, ["Take a moment to reflect."]))


# --- REPLACE your existing send_reset_email function with this one ---
# --- REPLACED send_reset_email function to use App Password ---
def send_reset_email(user):
    # Retrieve sender email and App Password from Flask config
    sender_email = app.config['MAIL_USERNAME']
    app_password = app.config['MAIL_PASSWORD'] # This should be your generated App Password

    # Ensure the App Password is set
    if not app_password:
        print("Error: MAIL_PASSWORD (App Password) not set in config. Cannot send email.")
        flash("Error sending password reset email. Please try again later.", "danger")
        return # Exit the function if no App Password

    try:
        # Generate the reset token
        token = user.get_reset_token()
        
        # Build the email message
        msg_flask = Message('Password Reset Request',
                            sender=sender_email,
                            recipients=[user.email])
        
        msg_flask.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
'''
        # Construct the email using MIMEText (or MIMEMultipart if you need HTML)
        mime_message = MIMEText(msg_flask.body)
        mime_message['to'] = ', '.join(msg_flask.recipients)
        mime_message['from'] = msg_flask.sender
        mime_message['subject'] = msg_flask.subject

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.set_debuglevel(0) # Set to 0 for production, 1 for debugging SMTP communication
        server.starttls() # Secure the connection
        
        # Authenticate using your Gmail address and App Password
        server.login(sender_email, app_password)
        
        # Send the email
        server.sendmail(msg_flask.sender, msg_flask.recipients, mime_message.as_string())
        server.quit()
        print(f"Email sent successfully to {user.email} using App Password!")
        flash('An email has been sent with instructions to reset your password.', 'info')

    except smtplib.SMTPAuthenticationError as e:
        print(f"SMTP Authentication Error: {e}")
        print("Please check your Gmail username and App Password. Ensure 2FA is enabled for your Gmail account and you are using an App Password, not your regular password.")
        flash("Authentication error: Could not send password reset email. Please check server configuration or contact support.", "danger")
    except Exception as e:
        print(f"Error sending email: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        flash("An unexpected error occurred while sending the password reset email. Please try again later.", "danger")


if __name__ == '__main__':
    app.run(debug=True)