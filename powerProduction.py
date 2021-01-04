
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
        X_train, X_test, y_train, y_test =super().trainTestSplit()
        self.kmeans.fit(X_train, y_train)
        labels = self.kmeans.fit_predict(X_train)
        df_with_Labels= df = pd.DataFrame(columns = ['Labels', 'X_train','y_train'])
        df['Labels'] = labels
        df['X_train'] = X_train
        df['y_train'] = y_train
        df_with_Labels.dropna()
        df_with_Labels_Cluster_0 = df_with_Labels[df_with_Labels['Labels'] == 0].dropna()
        df_with_Labels_Cluster_1 = df_with_Labels[df_with_Labels['Labels'] == 1].dropna()
        df_with_Labels_Cluster_2 = df_with_Labels[df_with_Labels['Labels'] == 2].dropna()
        predicted_label = self.kmeans.predict([[value]])
        if predicted_label == 0:
           
           self.reg.fit(df_with_Labels_Cluster_0[['X_train']],df_with_Labels_Cluster_0['y_train'])
           pickle.dump(self.reg, open('linearRegressionCluster0.pkl','wb'))
        elif predicted_label == 1:
             self.reg.fit(df_with_Labels_Cluster_1[['X_train']],df_with_Labels_Cluster_1['y_train'])
             pickle.dump(self.reg, open('linearRegressionCluster1.pkl','wb'))
        else:
             self.reg.fit(df_with_Labels_Cluster_2[['X_train']],df_with_Labels_Cluster_2['y_train'])
             pickle.dump(self.reg, open('linearRegressionCluster2.pkl','wb'))
        
        
        
        
         




powerproductionLinearRegression = PowerproductionLinearRegression()
powerproductionLinearRegression.linearRegression()
powerproductionKmeans = PowerproductionKmeans()


















