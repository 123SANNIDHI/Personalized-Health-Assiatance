from flask_login import UserMixin
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False) # Added email field
    password = db.Column(db.String(60), nullable=False)
    moods = db.relationship('Mood', backref='user', lazy=True)
    journal_entries = db.relationship('JournalEntry', backref='user', lazy=True)
    medications = db.relationship('Medication', backref='user', lazy=True)

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