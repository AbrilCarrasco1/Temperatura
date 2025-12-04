import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de temperatura ''')
st.image("temperatura.jpg", caption="Predicción de temperatura.")

st.header('Datos')

def user_input_features():
  # Entrada
  City = st.number_input('City', min_value=0, max_value=2, value = 0, step = 1)
  Year = st.number_input('Year',  min_value=0, max_value=3000, value = 0, step = 1)
  Month = st.number_input('Month', min_value=0, max_value=12, value = 0, step = 1)



  user_input_data = {'Year': Year,
                     'City': City,
                     'Month': Month,
                    }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

prediccion=0
datos =  pd.read_csv('Average_df.csv', encoding='latin-1')
X = datos.drop(columns='AverageTemperature')
y = datos['AverageTemperature']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1614954)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['City'] + b1[1]*df['Year'] + b1[2]*df['Month']

st.subheader('Cálculo de la temperatura')
st.write('La temperatura es: ', prediccion)
