import requests

# URL of your Flask app
url = 'http://51.103.209.30:5000/predict_number'

# Path to the photo you want to use for prediction
photo_path = "D:\Third_Year\\Neural\project\\fonts-dataset\IBM Plex Sans Arabic\\481.jpeg"

# JSON data to send in the POST request
data = {'photo_path': photo_path}

# Make the POST request
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Get the prediction from the response
    prediction = response.json()['prediction']
    print('Prediction:', prediction)
else:
    print('Error:', response.json()['error'])