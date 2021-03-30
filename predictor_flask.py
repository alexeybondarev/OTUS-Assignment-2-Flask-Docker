import pickle
from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix
import numpy as np

model = pickle.load(open('model_assignment_2.pkl', 'rb'))

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='OTUS Assignment_2 Bondarev', validate=True)
ns = api.namespace('heart', description='heart disease model')

heart_row = api.model('HeartRow', {
    'age': fields.Integer(required=True),
    'sex': fields.Integer(required=True),
    'cp': fields.Integer(required=True),
    'trestbps': fields.Integer(required=True),
    'chol': fields.Integer(required=True),
    'fbs': fields.Integer(required=True),
    'restecg': fields.Integer(required=True),
    'thalach': fields.Integer(required=True),
    'exang': fields.Integer(required=True),
    'oldpeak': fields.Integer(required=True),
    'slope': fields.Integer(required=True),
    'ca': fields.Integer(required=True),
    'thal': fields.Integer(required=True)
})

heart_prediction = api.inherit('HeartPrediction', heart_row, {
    'prediction': fields.List(fields.Float, min_items=2, max_items=2),
    'prediction_class': fields.Integer
})

@ns.route('/')
class HeartClassification(Resource):
    @ns.doc('obtain_prediction')
    @ns.expect(heart_row)
    @ns.marshal_with(heart_prediction, code=200)
    def post(self):
        payload = api.payload
        values_tuple = tuple(payload.values())
        prediction = [round(p, 5) for p in model.predict_proba([values_tuple])[0]]
        predicted_class = np.argmax(prediction)
        payload.update({'prediction': prediction})
        payload.update({'prediction_class': predicted_class})
        return payload

if __name__ == '__main__':
    app.run(debug=True)