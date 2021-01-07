import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
import pickle
class Powerproduction:
  def __init__(self):
    self.df =  pd.read_csv("powerproductionDataSet.csv")

  def trainTestSplit(self):
      X_train, X_test, y_train, y_test = train_test_split(self.df[['speed']], self.df['power'], test_size=0.2, random_state=0)
      return X_train, X_test, y_train, y_test





class PowerproductionLinearRegression(Powerproduction):
    def __init__(self):
        Powerproduction.__init__(self)
        self.reg = LinearRegression()
        
    def linearRegression(self):
        X_train, X_test, y_train, y_test =super().trainTestSplit()
        self.reg.fit(X_train, y_train)
        pickle.dump(self.reg, open('linearRegression.pkl','wb'))
        
    
        
      


class PowerproductionKmeans(Powerproduction):
    def __init__(self):
        Powerproduction.__init__(self)
        self.kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        self.reg = LinearRegression()
        
    def kMeans(self,value):
        self.kmeans.fit(self.df)
        y_kmeans = self.kmeans.fit_predict(self.df)
        cluster_2df = self.df.loc[y_kmeans == 2]
        X_train2, X_test2, y_train2, y_test2 = train_test_split(cluster_2df[['speed']], cluster_2df['power'], test_size=0.2, random_state=0)
        self.reg.fit(X_train2, y_train2)
        pickle.dump(self.reg, open('linearRegressionCluster2.pkl','wb'))
               
        
        
        
         
class NeuralNetworkTensorFlow(Powerproduction):
    def __init__(self):
        Powerproduction.__init__(self)
        
        
    def tensorFlow(self,value):
        train_dataset = self.df.sample(frac=0.8, random_state=0)
        test_dataset = self.df.drop(train_dataset.index)
        train_features = train_dataset.copy()
        test_features = test_dataset.copy()
        train_labels = train_features.pop('power')
        test_labels = test_features.pop('power')
        speed = np.array(train_features['speed'])
        speed_normalizer = preprocessing.Normalization(input_shape=[1,])
        speed_normalizer.adapt(speed)
        speed_model = tf.keras.Sequential([
        speed_normalizer,
        tf.keras.layers.Dense(50, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform")])
        speed_model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
        speed_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),loss='mean_absolute_error')
        speed_model.fit(train_features['speed'], train_labels,epochs=100,verbose=0,validation_split = 0.2)
        return speed_model.predict([value])



powerproductionLinearRegression = PowerproductionLinearRegression()
powerproductionLinearRegression.linearRegression()
powerproductionKmeans = PowerproductionKmeans()
neuralNetworkTensorFlow=NeuralNetworkTensorFlow()
print(neuralNetworkTensorFlow.tensorFlow(12.5))






