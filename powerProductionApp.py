# flask for web app.
import flask as fl
# numpy for numerical work.
from powerProduction import powerproductionLinearRegression
from powerProduction import powerproductionKmeans

# Create a new web app.
app = fl.Flask(__name__)

# Add root route.
@app.route("/")
def home():
  return app.send_static_file('index.html')

# Add linerRegression route.
@app.route('/predict/linerRegression/<int:value>')
def linerRegression():
  return {"value": powerproductionLinearRegression.linearRegression(value)}

# Add Kmeans Clustering sklearn route.
@app.route('/predict/Kmeans/<int:value>')
def kmeans():
  return {"value": powerproductionKmeans.kMeans(value)}
  
  
  
@app.route('/predict/tensorFlowKmeans')
def tensorFlowKmeans():
  return {"value": powerproductionKmeans.kMeans(value)}
  

@app.route('/predict/neutronFlowKmeans')
def tensorFlowneutron():
  return {"value": powerproductionKmeans.kMeans(value)}
  
  
if __name__ == "__main__":
    app.run(debug=True)