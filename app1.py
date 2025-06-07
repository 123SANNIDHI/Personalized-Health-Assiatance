# main.py
import base64
import os
import pickle
import smtplib
from email.mime.text import MIMEText
import secrets
from flask import Flask, request, render_template, redirect, url_for, session, flash
import numpy as np
import re
import pandas as pd
import pickle
import json
import random
from flask_mail import Mail
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Message
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, DateTimeField, TextAreaField, DateTimeLocalField, SelectField, RadioField, IntegerField,EmailField
from wtforms.validators import InputRequired, Length, ValidationError, DataRequired, EqualTo,Regexp,Email # Import EqualTo
from flask_bcrypt import Bcrypt
from datetime import datetime
from wtforms.validators import NumberRange
import os
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask import Flask, request, jsonify, render_template  # Import necessary Flask modules
from backend import initialize_llm, load_or_create_vector_db, setup_qa_chain
from itsdangerous import URLSafeTimedSerializer as Serializer
from flask_mail import Message

app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # required for CSRF protection

# Get absolute path to the DB file
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'database.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SECRET_KEY'] = 'thisisme'
3

# Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER') 
app.config['MAIL_USERNAME'] = 'sannidhiyk1@gmail.com'
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS')
mail = Mail(app)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize chatbot components when the Flask app starts
llm = initialize_llm()
vector_db = load_or_create_vector_db()
qa_chain = setup_qa_chain(vector_db, llm)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
# main.py
# ... other imports and configurations ...

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False) # Added email field
    password = db.Column(db.String(60), nullable=False)
    moods = db.relationship('Mood', backref='user', lazy=True)
    journal_entries = db.relationship('JournalEntry', backref='user', lazy=True)
    medications = db.relationship('Medication', backref='user', lazy=True)

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

class StrongPassword:
    """
    Custom validator to ensure a password meets complexity requirements.
    """
    def __init__(self, message=None):
        if not message:
            message = ('Password must contain at least one uppercase letter, '
                       'one lowercase letter, one number, and one special character.')
        self.message = message
        self.regex = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"

    def __call__(self, form, field):
        if not Regexp(self.regex)(form, field):
            raise ValidationError(self.message)
        
def ends_with_gmail(form, field):
    if not field.data.lower().endswith('@gmail.com'):
        raise ValidationError('Email address must end with "@gmail.com".')



class Mood(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    happy = db.Column(db.Integer)
    sad = db.Column(db.Integer)
    anxious = db.Column(db.Integer)
    angry = db.Column(db.Integer)
    calm = db.Column(db.Integer)
    energetic = db.Column(db.Integer)
    tired = db.Column(db.Integer)
    focused = db.Column(db.Integer)
    predominant_feeling = db.Column(db.String(50))
    other_feeling = db.Column(db.Text)
    stress_level = db.Column(db.String(20))
    physical_comfort = db.Column(db.String(20))
    social_interaction = db.Column(db.String(50))
    inferred_mood = db.Column(db.String(50)) # To store the overall inferred mood


class Medication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    label_name = db.Column(db.String(100))  # Optional label for the medication
    dosage_mrng = db.Column(db.String(50))
    dosage_aftn = db.Column(db.String(50))
    dosage_evng = db.Column(db.String(50))
    notes = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class QueryHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # Assuming you have a User model
    symptoms = db.Column(db.String(500))
    prediction = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    entry_text = db.Column(db.Text, nullable=False)

class AddMedicationForm(FlaskForm):
    name = StringField('Disease Name', validators=[InputRequired()])
    label_name = StringField('Medication Name', validators=[DataRequired()])
    dosage_mrng = StringField('Morning Dosage (e.g., 1 tablet)')
    dosage_aftn = StringField('Afternoon Dosage (e.g., 5mg)')
    dosage_evng = StringField('Evening Dosage (e.g., 1/2 tablet)')
    notes = TextAreaField('Notes')
    submit = SubmitField('Add Medication')

class EditMedicationForm(FlaskForm):
    name = StringField('Disease Name', validators=[InputRequired()])
    label_name = StringField('Medication Label (Optional)')
    dosage_mrng = StringField('Morning Dosage (e.g., 1 tablet)')
    dosage_aftn = StringField('Afternoon Dosage (e.g., 5mg)')
    dosage_evng = StringField('Evening Dosage (e.g., 1/2 tablet)')
    notes = TextAreaField('Notes')
    submit = SubmitField('Update Medication')

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

class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    email = EmailField('Email Address', validators=[InputRequired(), Email(),ends_with_gmail], render_kw={"placeholder": "Email Address"}) # Added EmailField
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20), EqualTo('confirm_password', message='Passwords must match'),StrongPassword()], render_kw={"placeholder": "Password"})
    confirm_password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Confirm Password"})  # Add confirm password field
    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError('That username already exists. Please choose a different one.')
    
    def validate_email(self, email): # Optional email validation
        existing_user_email = User.query.filter_by(email=email.data).first()
        if existing_user_email:
            raise ValidationError('That email address is already registered.')

class RequestResetForm(FlaskForm):
    email = EmailField('Email Address', validators=[InputRequired(), Email(),ends_with_gmail], render_kw={"placeholder": "Email Address"})
    submit = SubmitField('Request Password Reset')

    def validate_email(self, email): # Optional email validation
        existing_user_email = User.query.filter_by(email=email.data).first()
        if existing_user_email is None:
            raise ValidationError('That is no account with that email address. Please register first.')

class ResetPasswordForm(FlaskForm):
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20), EqualTo('confirm_password', message='Passwords must match'),StrongPassword()], render_kw={"placeholder": "Password"})
    confirm_password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Confirm Password"}) 
    submit = SubmitField('Reset Password')
        
@app.route('/Register', methods=['GET', 'POST'])
def Register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data,email=form.email.data,password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')  # Add success message
        return redirect(url_for('login'))
    return render_template('Register.html', form=form)  # No error variable needed, form.errors is used

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('login')



@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                flash('Login successful!', 'success')
                login_user(user)
                return redirect(url_for('sidebar'))
            else:
                flash('Invalid username or password', 'danger')  # Add flash message for wrong password
        else:
            flash('Invalid username or password', 'danger')  # Add flash message for user not found
    return render_template('login.html', form=form)  # No error variable needed, flash is used


@app.route('/medications')
@login_required
def medications():
    medications = current_user.medications
    form = AddMedicationForm()  # Make sure this line is present
    return render_template('medications.html', medications=medications, form=form)

@app.route('/add_medication', methods=['GET', 'POST'])
@login_required
def add_medication():
    form = AddMedicationForm()
    if form.validate_on_submit():
        new_medication = Medication(
            name=form.name.data,
            label_name=form.label_name.data,
            dosage_mrng=form.dosage_mrng.data,
            dosage_aftn=form.dosage_aftn.data,
            dosage_evng=form.dosage_evng.data,
            notes=form.notes.data,
            user_id=current_user.id
        )
        db.session.add(new_medication)
        db.session.commit()
        flash('Medication added successfully!', 'success')
        return redirect(url_for('medications'))

    return render_template('add_medication.html', form=form)

@app.route('/edit_medication/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_medication(id):
    medication = Medication.query.get_or_404(id)
    if medication.user_id != current_user.id:
        # Handle unauthorized access
        flash('You do not have permission to edit this medication.', 'danger')
        return redirect(url_for('medications'))
    form = EditMedicationForm(obj=medication)
    if form.validate_on_submit():
        medication.name = form.name.data
        medication.label_name = form.label_name.data
        medication.dosage_mrng = form.dosage_mrng.data
        medication.dosage_aftn = form.dosage_aftn.data
        medication.dosage_evng = form.dosage_evng.data
        medication.notes = form.notes.data
        db.session.commit()
        flash('Medication updated successfully!', 'success')
        return redirect(url_for('medications'))
    return render_template('edit_medication.html', form=form, medication_id=id, medication_name=medication.name)

@app.route('/update_medication/<int:id>', methods=['POST'])
@login_required
def update_medication(id):
    medication = Medication.query.get_or_404(id)
    if medication.user_id != current_user.id:
        flash('You do not have permission to update this medication.', 'danger')
        return redirect(url_for('medications'))
    form = EditMedicationForm()
    if form.validate_on_submit():
        medication.name = form.name.data
        medication.label_name = form.label_name.data
        medication.dosage_mrng = form.dosage_mrng.data
        medication.dosage_aftn = form.dosage_aftn.data
        medication.dosage_evng = form.dosage_evng.data
        medication.notes = form.notes.data
        db.session.commit()
        flash('Medication updated successfully!', 'success')
        return redirect(url_for('medications'))
    return redirect(url_for('edit_medication', id=id))  # Redirect back if not valid
@app.route('/delete_medication/<int:id>', methods=['POST'])
@login_required
def delete_medication(id):
    medication = Medication.query.get_or_404(id)
    if medication.user_id != current_user.id:
        flash('You do not have permission to delete this medication.', 'danger')
        return redirect(url_for('medications'))
    db.session.delete(medication)
    db.session.commit()
    flash('Medication deleted successfully!', 'success')
    return redirect(url_for('medications'))


@app.route('/sidebar')
@login_required
def sidebar():
    return render_template('sidebar.html', name=current_user.username)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
@login_required
def index():
    return render_template('index.html', name=current_user.username)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html') # Assuming you have an index.html with the chat interface

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')
    if not message:
        return jsonify({'error': 'Missing message'}), 400
    response = qa_chain.run(message)
    return jsonify({'response': response})

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        symptoms_str = request.form.get('symptoms', '').lower()
        parts = [s.strip() for s in symptoms_str.split('and')]
        user_symptoms = []
        for part in parts:
            user_symptoms.extend([s.strip() for s in part.split(',')])
        user_symptoms = [s.strip("[]' ") for s in user_symptoms if s]

        predicted_disease, matched_symptoms = get_predicted_value(
            user_symptoms, loaded_model, scaler_train, loaded_label_encoder, symptoms_dict, num_total_features
        )
        desc, pre, med, die, wrkout = helper(predicted_disease)

        # Save prediction history after prediction is complete
        history = QueryHistory(
            user_id=current_user.id,
            symptoms=",".join(user_symptoms),
            prediction=predicted_disease
        )
        db.session.add(history)
        db.session.commit()

        return render_template('index.html', predicted_disease=predicted_disease, dis_des=desc, my_precautions=pre, medications=med, my_diet=die, workout=wrkout, input_symptoms=matched_symptoms)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"An error occurred during prediction: {e}")

class MoodAssessmentForm(FlaskForm):
    happy = IntegerField('Rate the intensity of feeling happy (0-5):', validators=[DataRequired(), NumberRange(min=0, max=5)])
    sad = IntegerField('Rate the intensity of feeling sad (0-5):', validators=[DataRequired(), NumberRange(min=0, max=5)])
    anxious = IntegerField('Rate the intensity of feeling anxious (0-5):', validators=[DataRequired(), NumberRange(min=0, max=5)])
    angry = IntegerField('Rate the intensity of feeling angry/irritable (0-5):', validators=[DataRequired(), NumberRange(min=0, max=5)])
    calm = IntegerField('Rate the intensity of feeling calm (0-5):', validators=[DataRequired(), NumberRange(min=0, max=5)])
    energetic = IntegerField('Rate the intensity of feeling energetic (0-5):', validators=[DataRequired(), NumberRange(min=0, max=5)])
    tired = IntegerField('Rate the intensity of feeling tired (0-5):', validators=[DataRequired(), NumberRange(min=0, max=5)])
    focused = IntegerField('Rate the intensity of feeling focused (0-5):', validators=[DataRequired(), NumberRange(min=0, max=5)])

    stress_level_choices = [('Very Low', 'Very Low'), ('Low', 'Low'), ('Moderate', 'Moderate'), ('High', 'High'), ('Very High', 'Very High')]
    stress_level = RadioField('Overall stress level today:', choices=stress_level_choices, validators=[DataRequired()])

    physical_comfort_choices = [('Very Comfortable', 'Very Comfortable'), ('Comfortable', 'Comfortable'), ('Neutral', 'Neutral'), ('Uncomfortable', 'Uncomfortable'), ('Very Uncomfortable', 'Very Uncomfortable')]
    physical_comfort = RadioField('Physical comfort level right now:', choices=physical_comfort_choices, validators=[DataRequired()])

    social_interaction_choices = [('Very Positive', 'Very Positive'), ('Positive', 'Positive'), ('Neutral', 'Neutral'), ('Negative', 'Negative'), ('Very Negative', 'Very Negative'), ('No significant social interaction today', 'No significant social interaction today')]
    social_interaction = RadioField('Quality of social interactions today:', choices=social_interaction_choices, validators=[DataRequired()])

    submit = SubmitField('Submit Mood')

class JournalForm(FlaskForm):
    entry_text = TextAreaField('Your Thoughts', validators=[InputRequired()])
    submit = SubmitField('Save Entry')

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


@app.route("/history")
@login_required
def history():
    user_history = QueryHistory.query.filter_by(user_id=current_user.id).all()
    return render_template("history.html", history=user_history)

@app.route('/track_mood', methods=['GET', 'POST'])
@login_required
def track_mood():
    form = MoodAssessmentForm()
    inferred_mood = None
    suggestion = None

    if form.validate_on_submit():
        answers = {
            'happy': form.happy.data,
            'sad': form.sad.data,
            'anxious': form.anxious.data,
            'angry': form.angry.data,
            'calm': form.calm.data,
            'energetic': form.energetic.data,
            'tired': form.tired.data,
            'focused': form.focused.data,
            'stress_level': form.stress_level.data,
            'physical_comfort': form.physical_comfort.data,
            'social_interaction': form.social_interaction.data,
        }
        inferred_mood = infer_mood_critically(answers)
        suggestion = get_mood_suggestion(inferred_mood, answers)
        new_mood = Mood(
            user_id=current_user.id,
            timestamp=datetime.utcnow(),
            happy=form.happy.data,
            sad=form.sad.data,
            anxious=form.anxious.data,
            angry=form.angry.data,
            calm=form.calm.data,
            energetic=form.energetic.data,
            tired=form.tired.data,
            focused=form.focused.data,
            stress_level=form.stress_level.data,
            physical_comfort=form.physical_comfort.data,
            social_interaction=form.social_interaction.data,
            inferred_mood=inferred_mood
        )
        db.session.add(new_mood)
        db.session.commit()
        flash(f'Your inferred mood is: {inferred_mood}! Suggestion: {suggestion}', 'success')
        return render_template('track_mood.html', form=form, inferred_mood=inferred_mood, suggestion=suggestion)

    return render_template('track_mood.html', form=form)

@app.route('/mood_history')
@login_required
def mood_history():
    moods = Mood.query.filter_by(user_id=current_user.id).order_by(Mood.timestamp.desc()).all()
    mood_data_by_date = {}
    for mood in moods:
        date_str = mood.timestamp.strftime('%Y-%m-%d')
        if date_str not in mood_data_by_date:
            mood_data_by_date[date_str] = {}
        mood_data_by_date[date_str][mood.inferred_mood] = mood_data_by_date[date_str].get(mood.inferred_mood, 0) + 1

    mood_summary = []
    mood_colors = {
        'Happy': 'rgba(75, 192, 192, 0.7)',
        'Happy and Energetic': 'rgba(0, 128, 0, 0.7)',
        'Happy and Calm': 'rgba(144, 238, 144, 0.7)',
        'Sad': 'rgba(54, 162, 235, 0.7)',
        'Sad and Tired': 'rgba(173, 216, 230, 0.7)',
        'Anxious': 'rgba(255, 99, 132, 0.7)',
        'Highly Anxious': 'rgba(139, 0, 0, 0.7)',
        'Irritable': 'rgba(255, 159, 64, 0.7)',
        'Highly Irritable': 'rgba(255, 69, 0, 0.7)',
        'Calm': 'rgba(220, 220, 220, 0.7)',
        'Calm and Focused': 'rgba(169, 169, 169, 0.7)',
        'Low Energy': 'rgba(255, 205, 86, 0.7)',
        'Stressed and Happy': 'rgba(255, 165, 0, 0.7)',
        'Stressed and Sad': 'rgba(128, 128, 0, 0.7)',
        'Stressed and Anxious': 'rgba(255, 0, 0, 0.7)',
        'Physically Uncomfortable and Happy': 'rgba(0, 255, 0, 0.7)',
        'Physically Uncomfortable and Sad': 'rgba(0, 0, 255, 0.7)',
        'Socially Affected and Happy': 'rgba(255, 255, 0, 0.7)',
        'Socially Affected and Sad': 'rgba(128, 0, 128, 0.7)',
        'Neutral': 'rgba(201, 203, 207, 0.7)',
        'Other': 'rgba(128, 128, 128, 0.7)'
    }

    for date, moods in mood_data_by_date.items():
        summary_text = f"On {date}, you felt mostly "
        dominant_mood = max(moods, key=moods.get, default="Neutral")
        summary_text += f"{dominant_mood}"
        other_moods = ", ".join([f"{count} times {m}" for m, count in moods.items() if m != dominant_mood])
        if other_moods:
            summary_text += f", also experiencing {other_moods}"
        mood_summary.append({'date': date, 'summary': summary_text, 'moods': moods, 'colors': mood_colors})

    return render_template('mood_history.html', mood_summary=mood_summary)

@app.route('/journal', methods=['GET', 'POST'])
@login_required
def journal():
    form = JournalForm()
    if form.validate_on_submit():
        new_entry = JournalEntry(user_id=current_user.id, entry_text=form.entry_text.data)
        db.session.add(new_entry)
        db.session.commit()
        flash('Journal entry saved!', 'success')
        return redirect(url_for('journal_history'))
    return render_template('journal.html', form=form)

@app.route('/journal_history')
@login_required
def journal_history():
    entries = JournalEntry.query.filter_by(user_id=current_user.id).order_by(JournalEntry.timestamp.desc()).all()
    return render_template('journal_history.html', entries=entries)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

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

@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title='Reset Password', form=form)


@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', title='Reset Password', form=form)


if __name__ == '__main__':
    app.run(debug=True)