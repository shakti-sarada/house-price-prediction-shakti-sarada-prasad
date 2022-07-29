import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


house_price = Flask(__name__)
model1 = pickle.load(open('house_price.pkl','rb')) 


@house_price.route('/')
def home():
  
    return render_template("index.html")
  
@house_price.route('/predict',methods=['GET'])

def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    SqFt = float(request.args.get('SqFt'))
    Bedrooms = float(request.args.get('Bedrooms'))
    Bathrooms = float(request.args.get('Bathrooms'))
    Offers = float(request.args.get('Offers'))
    Brick = float(request.args.get('Brick'))
    Neighborhood = float(request.args.get('Neighborhood'))

    
    prediction = model1.predict([[SqFt,Bedrooms,Bathrooms,Offers,Brick,Neighborhood]])
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted House Price for given features is : {}'.format(prediction))


house_price.run()