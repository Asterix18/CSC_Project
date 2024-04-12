import pandas as pd
import numpy as np
from flask import Flask, flash, render_template, request, jsonify, redirect, url_for, session, make_response
from joblib import load
from functools import wraps
from dateutil import relativedelta
from datetime import datetime


def create_app():
    app = Flask(__name__)

    app.secret_key = "SECRET"

    # Set up logins
    logins = {"Mike": "password1", "John": "password2", "admin": "admin"}

    # Load machine learning model
    five_year = load('Models/5yr_model.joblib')
    ten_year = load('Models/10yr_model.joblib')

    # Define the feature names expected by model
    five_year_feature_names = ['age_at_diagnosis_in_years', 'tnm_stage', 'mmr_status',
                               'rfs_event_censored_5yr', 'sex_Male', 'tumour_location_proximal',
                               'chemotherapy_adjuvant_Y',
                               'kras_mutation_WT']

    ten_year_feature_names = ['age_at_diagnosis_in_years', 'tnm_stage', 'chemotherapy_adjuvant_Y', 'CMS',
                              'rfs_event_censored_5yr', 'kras_mutation_WT']

    # Function to delete cache to avoid unauthorised access
    def nocache(view):
        @wraps(view)
        def no_cache(*args, **kwargs):
            response = make_response(view(*args, **kwargs))
            response.headers[
                'Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '-1'
            return response

        return no_cache

    # Function to check logged in
    def logged_in():
        return session.get('logged_in')

    @app.route('/')
    def home():
        return redirect(url_for('login'))

    # Reroute unauthenticated users
    @app.route('/unauthenticated')
    def unauthenticated():
        return render_template('unauthenticated.html')

    # Route use to login page
    @app.route('/login', methods=['GET', 'POST'])
    @nocache
    def login():
        if logged_in():
            return redirect(url_for('form'))
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if username in logins and logins[username] == password:
                session['logged_in'] = True
                return redirect(url_for('form'))
            else:
                flash('Invalid Credentials. Please try again.')
        return render_template('login.html')

    # Log out user
    @app.route('/logout')
    def logout():
        if not logged_in():
            flash('Please login to continue.')
            return redirect(url_for('unauthenticated'))
        session.pop('logged_in', None)
        flash('You have successfully logged out.')
        return redirect(url_for('unauthenticated'))

    # Load form
    @app.route('/form')
    @nocache
    def form():
        if not logged_in():
            flash('Please login to continue.')
            return redirect(url_for('unauthenticated'))
        # Load index html
        return render_template('index.html')

    # Calculate probabilities
    @app.route('/predict', methods=['POST'])
    @nocache
    def predict():
        if not logged_in():
            flash('Please login to continue.')
            return redirect(url_for('unauthenticated'))
        try:
            # Extract and process form data
            form_data = request.form
            first_name = str(form_data['forename'])
            last_name = str(form_data['surname'])
            dob = datetime.strptime(str(form_data['age']), "%Y-%m-%d")
            tnm_stage = int(form_data['tnm_stage'])
            mmr_status = int(form_data['mmr_status'])
            rfs_event_censored_5yr = int(form_data['rfs_event_censored_5yr'])
            sex_Male = form_data['sex_Male'] == 'True'
            tumour_location_proximal = form_data['tumour_location_proximal'] == 'True'
            chemotherapy_adjuvant_Y = form_data['chemotherapy_adjuvant_Y'] == 'True'
            kras_mutation_WT = form_data['kras_mutation_WT'] == 'True'
            cms = int(form_data['cms'])

            # Convert dob to age here
            # Current date
            current_date = datetime.now()

            # Calculate the difference in years between current date and dob
            age_difference = relativedelta.relativedelta(current_date, dob)
            age = age_difference.years

            # Five Year Calculations
            five_year_input_data = [[age, tnm_stage, mmr_status, rfs_event_censored_5yr, sex_Male,
                                     tumour_location_proximal, chemotherapy_adjuvant_Y, kras_mutation_WT]]

            five_year_input_df = pd.DataFrame(five_year_input_data, columns=five_year_feature_names)

            # Create 5 year survival function
            five_year_survival_probabilities = five_year.predict_survival_function(five_year_input_df)[0]
            five_year_time_points = five_year_survival_probabilities.x
            five_year_probabilities = five_year_survival_probabilities.y

            # Extract 1 and 5 year probabilities
            one_year_probability = five_year_probabilities[np.where(five_year_time_points == 12)[0][0]]
            five_year_probability = five_year_probabilities[np.where(five_year_time_points == 60)[0][0]]

            # Ten Year Calculations
            ten_year_input_data = [
                [age, tnm_stage, chemotherapy_adjuvant_Y, cms, rfs_event_censored_5yr, kras_mutation_WT]]

            ten_year_input_df = pd.DataFrame(ten_year_input_data, columns=ten_year_feature_names)

            ten_year_survival_probabilities = ten_year.predict_survival_function(ten_year_input_df)[0]
            ten_year_time_points = ten_year_survival_probabilities.x
            ten_year_probabilities = ten_year_survival_probabilities.y

            # Extract 10 year probability
            ten_year_probability = ten_year_probabilities[np.where(ten_year_time_points == 120)[0][0]]

            return jsonify(one_year=one_year_probability, five_year=five_year_probability,
                           ten_year=ten_year_probability,
                           first_name=first_name, last_name=last_name)
        except Exception as e:
            print(e)
            return jsonify(error="An error occurred during prediction. Please try again."), 400

    # Route to about page
    @app.route('/about')
    @nocache
    def about():
        if not logged_in():
            flash('Please login to continue.')
            return redirect(url_for('unauthenticated'))
        return render_template("about.html")

    # Route to saved predictions page
    @app.route('/savedPredictions')
    @nocache
    def savedPredictions():
        if not logged_in():
            flash('Please login to continue.')
            return redirect(url_for('unauthenticated'))
        return render_template('savedPredictions.html')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)