from flask import Flask,jsonify,request
import numpy as np
import sys
import pickle
from flask_cors import CORS
import sklearn

app = Flask(__name__)
CORS(app)

model = pickle.load(open('model_1.pkl','rb'))
print(model)
# classifier = pickle.load(open('model.pkl', 'rb'))

@app.route("/", methods=['GET'])
def home():
    return jsonify({
        "result":"Hello",
    })

@app.route("/predict", methods=['POST'])
def predict():
		try: 
			formData = request.get_json(force=True)
			data = [val for val in formData.values()]
			prediction = model.predict(np.array(data).reshape(1, -1))
			types = { 0: "No", 1: "Yes "}
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": "Attrition "+ types[prediction[0]]
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})
