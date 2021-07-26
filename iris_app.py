import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
iris_df = pd.read_csv("iris-species.csv")
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(X_train, y_train)
log_reg = LogisticRegression(n_jobs = -1)
log_reg.fit(X_train, y_train)
@st.cache()
def prediction(model,sl,sw,pl,pw):
  predict=model.predict([[sl,sw,pl,pw]])
  predicted=predict[0]
  if predicted==0:
    return 'Iris-setosa'
  elif predicted==1:
    return 'Iris-virginica'
  else:
    return 'Iris-versicolor'
st.sidebar.title('Iris flower prediction model')
sl=st.sidebar.slider('Sepal Length: ',float(iris_df['SepalLengthCm'].min()),float(iris_df['SepalLengthCm'].max()))
sw=st.sidebar.slider('Sepal Width: ',float(iris_df['SepalWidthCm'].min()),float(iris_df['SepalWidthCm'].max()))
pl=st.sidebar.slider('Petal Length: ',float(iris_df['PetalLengthCm'].min()),float(iris_df['PetalLengthCm'].max()))
pw=st.sidebar.slider('Petal Width: ',float(iris_df['PetalWidthCm'].min()),float(iris_df['PetalWidthCm'].max()))
m=st.sidebar.selectbox('Classifier',('SVC','Logistic Regression','Random Forest Classifier'))
if st.sidebar.button('Predict'):
  if m=='SVC':
    pred=prediction(svc_model,sl,sw,pl,pw)
    score=svc_model.score(X_train,y_train)
  elif m=='Logistic Regression':
    pred=prediction(log_reg,sl,sw,pl,pw)
    score=log_reg.score(X_train,y_train)
  else:
    pred=prediction(rf_clf,sl,sw,pl,pw)
    score=rf_clf.score(X_train,y_train)  
  st.write('The species has been predicted as: ',pred)
  st.write('The accuracy of the prediction is: ',score)
