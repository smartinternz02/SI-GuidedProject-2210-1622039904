import requests

API_KEY = "M87sApAccVXVuRRvEvTyaR4NvIAUwM5EGVx_7mK8UOoV"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

payload_scoring = {"input_data": [{"field": ['avg_air_flow_267', 'avg_float_level_47',
                                             '%Iron Feed',	'Amina Flow',	'Ore Pulp pH',	'Ore Pulp Density', '% Silica Concentrate'],
 "values": [[251.44800,
 483.4510,
 55.2,
 557.434,
 10.0664,
 1.74]]}]}

response_scoring = requests.post(r"https://us-south.ml.cloud.ibm.com/ml/v4/deployments/f4ca803e-2b02-4028-a946-2def69acf066/predictions?version=2021-06-04", json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
print(response_scoring.json())




import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model= pickle.load(open('mining.pkl','rb'))

              
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/about')
def about():
    return  render_template("about.html")
@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    
    x_test = [[x for x in request.form.values()]]
    prediction = model.predict(x_test)
    pred=prediction[0]
    print(prediction)
 
    return render_template('index.html', prediction_text='Predicted Quality:{}'.format(pred))

if __name__ == "__main__":
    app.run(debug=True)
