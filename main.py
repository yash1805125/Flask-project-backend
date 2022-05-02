from flask import Flask, jsonify, request, make_response, render_template
from flask_restful import Resource, Api
from flask_cors import CORS
import numpy as np
# import sys
import pickle

app = Flask(__name__)
api = Api(app)
CORS(app)

classifier = pickle.load(open('model.pkl', 'rb'))


class PredictSingle(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response
		
	def post(self):
		try: 
			formData = request.get_json(force=True)
			data = [val for val in formData.values()]
			prediction = classifier.predict(np.array(data).reshape(1, -1))
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

class PredictExcel(Resource):
	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	def post(self):
		try:
			formData = request.json
			data = [val for val in formData.values()]
			prediction = classifier.predict(np.array(data).reshape(1, -1))
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


class Home(Resource):    
     def get(self):
         try:
            return make_response(render_template('index.html'))
         except(error): 
            return {'data': error}

api.add_resource(Home, '/')
api.add_resource(PredictSingle, '/prediction')
api.add_resource(PredictExcel,'/excelpred')

if __name__ == '__main__':
    app.run(debug=True)