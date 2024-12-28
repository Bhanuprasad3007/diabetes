import os
import pickle
import sqlite3

import numpy as np
from flask import (Flask, jsonify, redirect, render_template, request, session,
                   url_for)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

# Load pre-trained models
with open('diabetes-prediction-model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Database setup
DB_NAME = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT,
                        password TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        input_data TEXT,
                        result TEXT)''')
    conn.commit()
    conn.close()

# Home route
@app.route('/')
def home():
        return render_template('home.html')
   

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return 'Invalid credentials'
    return render_template('login.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')

# Diabetes Prediction Route
@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        try:
            # Get inputs from the form
            age = int(request.form['Age'])
            bmi = float(request.form['BMI'])
            insulin = int(request.form['Insulin'])
            glucose = int(request.form['Glucose'])
            family_history = request.form['FamilyHistory']

            # Clean the family_history input (remove extra spaces and convert to lowercase)
            family_history = family_history.strip().lower()

            # Check if 'family_history' is valid
            if family_history not in label_encoder.classes_:
                return f"<h1>Error: Unrecognized family history value: {family_history}</h1>"

            # Encode 'FamilyHistory'
            family_history_numeric = label_encoder.transform([family_history])[0]

            # Prepare input data
            input_data = np.array([[age, bmi, insulin, glucose, family_history_numeric]])

            # Prediction
            prediction = model.predict(input_data)[0]

            # Interpret result
            if prediction == 0:
                result = 'No Diabetes'
            elif prediction == 1:
                result = 'Type 1 Diabetes'
            elif prediction == 2:
                result = 'Type 2 Diabetes'
            else:
                result = 'Unknown Result'

            # Save result in history if user is logged in
            if 'user_id' in session:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('INSERT INTO history (user_id, input_data, result) VALUES (?, ?, ?)', 
                               (session['user_id'], str(input_data.tolist()), result))
                conn.commit()
                conn.close()

            return render_template('result.html', result=result)

        except Exception as e:
            return f'<h1>Error: {str(e)}</h1>'

    return render_template('test.html')


# Additional Routes
@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/medication')
def medication():
    return render_template('medication.html')

@app.route('/exercise')
def exercise():
    return render_template('exercise.html')

@app.route('/account')
def account():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Fetch history data from database
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT input_data, result FROM history WHERE user_id = ?', (session['user_id'],))
        history = cursor.fetchall()

    return render_template('account.html', username=session['username'], history=history)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Initialize database and run app
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
