
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from prediction import get_prediction


st.set_page_config(page_title='Site Energy Intensity Prediction', page_icon="⚡",
                               layout="wide", initial_sidebar_state='expanded')

pickle_in = open('model.pkl', 'rb') 
model = pickle.load(pickle_in)


# creating option list for dropdown menu

features = ['february_avg_temp', 'march_avg_temp', 'december_avg_temp', 'avg_temp',
       'month_heating_degree_days']

st.markdown("<h1 style='text-align: center;'>Site Energy Intensity Prediction App ⚡ </h1>", unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):

       st.header("Predict the input for following features:")

       february_avg_temp = st.slider('february_avg_temp', 20.00, 60.00, value=20.00, format="%f")
       march_avg_temp = st.slider('march_avg_temp', 30.00, 70.00 , value=30.00, format="%f")
       december_avg_temp = st.slider('december_avg_temp', 30.00, 60.00 , value=30.00, format="%f")
       avg_temp = st.slider('avg_temp', 50.000, 80.000 , value=50.000, format="%f")
       month_heating_degree_days = st.slider( 'month_heating_degree_days', 150.00, 500.00, value=150.00, format="%f")
       submit = st.form_submit_button("Predict")

    if submit:
 
       data = np.array([february_avg_temp, march_avg_temp, december_avg_temp, avg_temp,
       month_heating_degree_days]).reshape(1, -1)
       pred = get_prediction(data=data, model=model)
       st.write(f"The predicted Site Energy Intensity is:  {pred}")


if __name__ == '__main__':
       main()




 

