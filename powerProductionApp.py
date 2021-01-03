# flask for web app.
from flask import Flask, request, jsonify, render_template
# numpy for numerical work.
from powerProduction import powerproductionLinearRegression
from powerProduction import powerproductionKmeans
import pickle
import os.path
# Create a new web app.
app = Flask(__name__)


#modelKmeans1 = pickle.load(open('linearRegressionCluster0.pkl', 'rb'))
#modelKmeans2 = pickle.load(open('linearRegressionCluster1.pkl', 'rb'))
#modelKmeans3 = pickle.load(open('linearRegressionCluster2.pkl', 'rb'))

# Add root route.
@app.route("/")
def home():
  return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    req = request.form
    valuetoPredict = float(req.get("speedToPredictPower"))
    machineLearningModel = req.get("prediction")
   
    if machineLearningModel == "Linear Regression":
         modelLinearRegression = pickle.load(open('linearRegression.pkl', 'rb'))
         prediction = modelLinearRegression.predict([[valuetoPredict]])
    elif machineLearningModel == "Sklearn Kmeans":
         if os.path.isfile('linearRegressionCluster0.pkl'):
             modelLinearRegression = pickle.load(open('linearRegressionCluster0.pkl', 'rb'))
             prediction = modelLinearRegression.predict([[valuetoPredict]])
         elif os.path.isfile('linearRegressionCluster1.pkl'):
            modelLinearRegression = pickle.load(open('linearRegressionCluster1.pkl', 'rb'))
            prediction = modelLinearRegression.predict([[valuetoPredict]])
         else:
           modelLinearRegression = pickle.load(open('linearRegressionCluster2.pkl', 'rb'))
           prediction = modelLinearRegression.predict([[valuetoPredict]])
    else:
        modelLinearRegression = pickle.load(open('linearRegressionCluster2.pkl', 'rb'))
        prediction = modelLinearRegression.predict([[valuetoPredict]])

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predict power in the turnbine is  $ {}'.format(output))
  
 

 
if __name__ == "__main__":
    app.run(debug=True)