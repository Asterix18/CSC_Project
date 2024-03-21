from flask import Flask, render_template, request, jsonify
from joblib import load
import pandas as pd
import numpy as np
from dateutil import relativedelta
from datetime import datetime

app = Flask(__name__)

# Load machine learning model
five_year = load('Models/5yr_model.joblib')
ten_year = load('Models/10yr_model.joblib')

# Define the feature names expected by model
five_year_feature_names = ['age_at_diagnosis_in_years', 'tnm_stage', 'mmr_status', 'cimp_status',
                           'rfs_event_censored_5yr', 'tumour_location_proximal', 'chemotherapy_adjuvant_Y',
                           'kras_mutation_WT', 'braf_mutation_WT']

ten_year_feature_names = ['age_at_diagnosis_in_years', 'tnm_stage', 'CMS',
                          'rfs_event_censored_10yr', 'sex_Male', 'tumour_location_proximal', 'chemotherapy_adjuvant_Y',
                          'tp53_mutation_WT', 'braf_mutation_WT']


@app.route('/')
def login():
    return render_template('index.html')


@app.route('/login')
def login_redirect():
    return render_template('login.html')


@app.route('/form')
def form():
    # This could be the page with the form if separate, or just redirect to home/login
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and process form data
        form_data = request.form
        first_name = str(form_data['forename'])
        last_name = str(form_data['surname'])
        dob = datetime.strptime(str(form_data['age']), "%Y-%m-%d")
        tnm_stage = int(form_data['tnm_stage'])
        mmr_status = int(form_data['mmr_status'])
        cimp_status = int(form_data['cimp_status'])
        rfs_event_censored_5yr = int(form_data['rfs_event_censored_5yr'])
        rfs_event_censored_10yr = int(form_data['rfs_event_censored_10yr'])
        sex_Male = form_data['sex_Male'] == 'True'
        tumour_location_proximal = form_data['tumour_location_proximal'] == 'True'
        chemotherapy_adjuvant_Y = form_data['chemotherapy_adjuvant_Y'] == 'True'
        kras_mutation_WT = form_data['kras_mutation_WT'] == 'True'
        braf_mutation_WT = form_data['braf_mutation_WT'] == 'True'
        tp53_mutation_WT = form_data['tp53_mutation_WT'] == 'True'
        cms = int(form_data['cms'])

        # Convert dob to age here
        # Current date
        current_date = datetime.now()

        # Calculate the difference in years between current date and dob
        age_difference = relativedelta.relativedelta(current_date, dob)
        age = age_difference.years

        # Five Year Calculations
        five_year_input_data = [[age, tnm_stage, mmr_status, cimp_status, rfs_event_censored_5yr,
                                 tumour_location_proximal, chemotherapy_adjuvant_Y, kras_mutation_WT, braf_mutation_WT]]

        five_year_input_df = pd.DataFrame(five_year_input_data, columns=five_year_feature_names)

        survival_probabilities = five_year.predict_survival_function(five_year_input_df)[0]
        five_year_time_points = survival_probabilities.x
        five_year_probabilities = survival_probabilities.y

        # Extract the survival probabilities at specific time points
        one_year_probability = five_year_probabilities[np.where(five_year_time_points == 12)[0][0]]
        five_year_probability = five_year_probabilities[np.where(five_year_time_points == 60)[0][0]]

        # Ten Year Calculations
        ten_year_input_data = [[age, tnm_stage, cms, rfs_event_censored_10yr, sex_Male, tumour_location_proximal,
                                chemotherapy_adjuvant_Y, tp53_mutation_WT, braf_mutation_WT]]

        ten_year_input_df = pd.DataFrame(ten_year_input_data, columns=ten_year_feature_names)

        survival_probabilities = ten_year.predict_survival_function(ten_year_input_df)[0]
        ten_year_time_points = survival_probabilities.x
        ten_year_probabilities = survival_probabilities.y

        # Extract the survival probabilities at specific time points
        ten_year_probability = ten_year_probabilities[np.where(ten_year_time_points == 120)[0][0]]

        # Return JSON data
        return jsonify(one_year=one_year_probability, five_year=five_year_probability, ten_year=ten_year_probability,
                       first_name=first_name, last_name=last_name)
    except Exception as e:
        print(e)
        return jsonify(error="An error occurred during prediction. Please try again."), 400


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == '__main__':
    app.run(debug=True)
