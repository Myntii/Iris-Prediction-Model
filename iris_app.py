import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
iris_df = pd.read_csv("https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/iris-species.csv")
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state = 42)
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)
@st.cache()
def prediction(sl,sw,pl,pw):
  predict=svc_model.predict([[sl,sw,pl,pw]])
  predicted=predict[0]
  if predicted==0:
    return 'Iris-setosa'
  elif predicted==1:
    return 'Iris-virginica'
  else:
    return 'Iris-versicolor'
st.title('Iris flower prediction model')
sl=st.slider('Sepal Length: ',0.0,10.0)
sw=st.slider('Sepal Width: ',0.0,10.0)
pl=st.slider('Petal Length: ',0.0,10.0)
pw=st.slider('Petal Width: ',0.0,10.0)
if st.button('Predict'):
  pred=prediction(sl,sw,pl,pw)
  st.write('The species has been predicted as: ',pred)
  st.write('The accuracy of the prediction is: ',score)