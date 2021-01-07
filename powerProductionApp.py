# flask for web app.
from flask import Flask, request, jsonify, render_template
# numpy for numerical work.
from powerProduction import powerproductionLinearRegression,powerproductionKmeans,neuralNetworkTensorFlow
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
    try:
       valuetoPredict = float(req.get("speedToPredictPower"))
       machineLearningModel = req.get("prediction")
       
       print(machineLearningModel + " model ")
       if machineLearningModel == "1":
         powerproductionLinearRegression.linearRegression()
         modelLinearRegression = pickle.load(open('linearRegression.pkl', 'rb'))
         prediction = modelLinearRegression.predict([[valuetoPredict]])
       elif machineLearningModel == "2":
          powerproductionKmeans.kMeans(valuetoPredict)
          modelLinearRegression = pickle.load(open('linearRegressionCluster2.pkl', 'rb'))
          prediction = modelLinearRegression.predict([[valuetoPredict]])
       else:
          prediction = neuralNetworkTensorFlow.tensorFlow(valuetoPredict)[0]

       output = round(prediction[0], 2)
    except ValueError:
       return render_template('index.html', error_text='Predicted value is requried ')   
    

    return render_template('index.html', prediction_text='Predict power using the method ' + machineLearningModel + '  in the turnbine is  $ {}'.format(output))
  
 

 
if __name__ == "__main__":
    app.run(debug=True)