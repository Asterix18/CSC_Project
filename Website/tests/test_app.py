import pytest
from Website.app import create_app
from flask import session, url_for


@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client


# Tests for login page
def test_login_page(client):
    response = client.get('/login')
    assert response.status_code == 200


def test_home(client):
    response = client.get('/', follow_redirects=True)
    assert response.status_code == 200
    assert response.request.path == url_for('login')


def test_valid_login(client):
    response = client.post('/login', data={'username': 'Mike', 'password': 'password1'}, follow_redirects=True)
    assert response.status_code == 200
    assert session.get('logged_in') is True


def test_invalid_login(client):
    response = client.post('/login', data={'username': 'Mike', 'password': 'wrongpassword'}, follow_redirects=True)
    assert b'Invalid Credentials. Please try again.' in response.data


def test_already_logged_in(client):
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    response = client.get('/login', follow_redirects=True)
    assert response.status_code == 200
    assert response.request.path == url_for('form')


# Tests for unauthenticated access to internal pages
def test_logout(client):
    response = client.get('/logout', follow_redirects=True)
    assert response.request.path == url_for('unauthenticated')
    assert response.status_code == 200


def test_unauthenticated(client):
    response = client.get('/unauthenticated')
    assert response.request.path == url_for('unauthenticated')
    assert response.status_code == 200


def test_unauthenticated_about(client):
    response = client.get('/about', follow_redirects=True)
    assert response.status_code == 200
    assert response.request.path == url_for('unauthenticated')
    assert b'Please login to continue.' in response.data


def test_unauthenticated_form(client):
    response = client.get('/form', follow_redirects=True)
    assert response.status_code == 200
    assert response.request.path == url_for('unauthenticated')
    assert b'Please login to continue.' in response.data


def test_unauthenticated_savedPredictions(client):
    response = client.get('/savedPredictions', follow_redirects=True)
    assert response.status_code == 200
    assert response.request.path == url_for('unauthenticated')
    assert b'Please login to continue.' in response.data


def test_unauthenticated_predict(client):
    response = client.post('/predict', follow_redirects=True)
    assert response.status_code == 200
    assert response.request.path == url_for('unauthenticated')
    assert b"Please login to continue." in response.data


# Tests for authenticated access to pages
def test_form(client):
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    response = client.get('/form')
    assert response.request.path == url_for('form')
    assert response.status_code == 200


def test_about(client):
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    response = client.get('/about')
    assert response.request.path == url_for('about')
    assert response.status_code == 200


def test_savedPredictions(client):
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    response = client.get('/savedPredictions')
    assert response.request.path == url_for('savedPredictions')
    assert response.status_code == 200


# Test logout functionality
def test_logout_functionality(client):
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    response = client.get('/logout', follow_redirects=True)
    assert response.status_code == 200
    assert response.request.path == url_for('unauthenticated')
    assert 'logged_in' not in session
    assert b'You have successfully logged out' in response.data


# End-to-end tests to test route user takes to get predictions
def test_predict_route(client):
    # Login
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    # Sample input test data
    test_input_data = {
        'forename': 'John',
        'surname': 'Doe',
        'age': '1980-01-01',
        'tnm_stage': '2',
        'mmr_status': '1',
        'rfs_event_censored_5yr': '1',
        'sex_Male': 'True',
        'tumour_location_proximal': 'True',
        'chemotherapy_adjuvant_Y': 'True',
        'kras_mutation_WT': 'True',
        'cms': '1'
    }

    # POST request to the predict endpoint
    response = client.post('/predict', data=test_input_data)

    # Get the JSON response
    json_data = response.get_json()

    assert response.status_code == 200

    # Check probabilities are returned
    assert 0 < json_data['one_year'] < 1
    assert 0 < json_data['five_year'] < 1
    assert 0 < json_data['ten_year'] < 1

    # Check names are returned
    assert json_data['first_name'] == 'John'
    assert json_data['last_name'] == 'Doe'


def test_predict_no_patient_data(client):
    # Login
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    # Sample input test data
    test_input_data = {
        'forename': 'John',
        'surname': 'Doe',
    }

    # POST request to the predict endpoint
    response = client.post('/predict', data=test_input_data)

    # Get the JSON response
    json_data = response.get_json()

    assert response.status_code == 400

    # Check names are returned
    assert json_data['error'] == 'An error occurred during prediction. Please try again.'


def test_predict_invalid_data_type(client):
    # Login
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    # Sample input test data
    test_input_data = {
        'forename': 'John',
        'surname': 'Doe',
        'age': 'Not an age',
        'tnm_stage': '2',
        'mmr_status': '1',
        'rfs_event_censored_5yr': '1',
        'sex_Male': 'True',
        'tumour_location_proximal': 'True',
        'chemotherapy_adjuvant_Y': 'True',
        'kras_mutation_WT': 'True',
        'cms': '1'
    }

    # POST request to the predict endpoint
    response = client.post('/predict', data=test_input_data)

    # Get the JSON response
    json_data = response.get_json()

    assert response.status_code == 400

    # Check names are returned
    assert json_data['error'] == 'An error occurred during prediction. Please try again.'


# Integration tests to check users cannot return to internal pages after logging out
def test_after_logout_form(client):
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    client.get('/logout')
    response = client.get('/form', follow_redirects=True)
    assert response.status_code == 200
    assert response.request.path == url_for('unauthenticated')
    assert b'Please login to continue.' in response.data


def test_after_logout_about(client):
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    client.get('/logout')
    response = client.get('/about', follow_redirects=True)
    assert response.status_code == 200
    assert response.request.path == url_for('unauthenticated')
    assert b'Please login to continue.' in response.data


def test_after_logout_savedPredictions(client):
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    client.get('/logout')
    response = client.get('/savedPredictions', follow_redirects=True)
    assert response.status_code == 200
    assert response.request.path == url_for('unauthenticated')
    assert b'Please login to continue.' in response.data


# Test to check a user making multiple requests
def test_multiple_requests(client):
    client.post('/login', data={'username': 'Mike', 'password': 'password1'})
    response1 = client.get('/form')
    response2 = client.get('/about')
    assert session.get('logged_in') is True
    assert response1.status_code == 200
    assert response2.status_code == 200

