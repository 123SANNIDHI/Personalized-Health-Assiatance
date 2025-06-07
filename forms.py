from flask_wtf import FlaskForm
from wtforms import (
    StringField, PasswordField, SubmitField, TextAreaField,
    IntegerField, RadioField, EmailField
)
from wtforms.validators import (
    DataRequired, Length, Email, EqualTo, ValidationError, Regexp, InputRequired, NumberRange
)

# Custom validator for strong password (as you provided)
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
        raise ValidationError('Email must end with @gmail.com')

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

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('login')



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



# --- NEW: Chat Form (primarily for backend validation if used) ---
class ChatForm(FlaskForm):
    message = StringField('Message', validators=[DataRequired(), Length(min=1, max=500)])
    submit = SubmitField('Send')