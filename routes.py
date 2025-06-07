import bcrypt
from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from app import app, db  # Make sure this matches your main file name!
from app import get_predicted_value, helper, symptoms_dict, num_total_features, loaded_model, scaler_train, loaded_label_encoder
from datetime import datetime
from app import infer_mood_critically, get_mood_suggestion
from forms import MoodAssessmentForm
from app import QueryHistory, Mood
from app import JournalEntry,Medication, Mood,User, send_reset_email
import forms


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

@app.route('/Register', methods=['GET', 'POST'])
def Register():
    form = forms.RegisterForm()  # Use RegisterForm from forms module
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')  # Add success message
        return redirect(url_for('login'))
    return render_template('Register.html', form=form)  # No error variable needed, form.errors is used



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
    """
    Render the home page of the application.
    """
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