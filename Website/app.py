from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model
rsf = load('Models/rsf_model.joblib')
feature_names = ['age_at_diagnosis_in_years', 'tnm_stage', 'mmr_status', 'cimp_status', 'rfs_event_censored_5yr', 'sex_Male', 'tumour_location_proximal', 'chemotherapy_adjuvant_Y', 'kras_mutation_WT']

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extract and process form data
    form_data = request.form
    age = float(form_data['age'])
    tnm_stage = form_data['tnm_stage']
    mmr_status = int(form_data['mmr_status'])
    cimp_status = int(form_data['cimp_status'])
    rfs_event_censored_5yr = int(form_data['rfs_event_censored_5yr'])
    sex_Male = form_data['sex_Male'] == 'True'
    tumour_location_proximal = form_data['tumour_location_proximal'] == 'True'
    chemotherapy_adjuvant_Y = form_data['chemotherapy_adjuvant_Y'] == 'True'
    kras_mutation_WT = form_data['kras_mutation_WT'] == 'True'

    input_data = [[age, tnm_stage, mmr_status, cimp_status, rfs_event_censored_5yr, sex_Male, tumour_location_proximal,
                   chemotherapy_adjuvant_Y, kras_mutation_WT]]

    # Assuming 'feature_names' is defined globally or loaded/defined before using it here
    input_df = pd.DataFrame(input_data, columns=feature_names)

    survival_probabilities = rsf.predict_survival_function(input_df)[0]
    time_points = survival_probabilities.x
    probabilities = survival_probabilities.y
    # Extract the survival probabilities at specific time points
    one_year = probabilities[np.where(time_points == 12)[0][0]]
    three_year = probabilities[np.where(time_points == 36)[0][0]]
    five_year = probabilities[np.where(time_points == 60)[0][0]]

    # Pass the predictions to a new template or return them directly
    return render_template('results.html', one_year=one_year, three_year=three_year, five_year=five_year)