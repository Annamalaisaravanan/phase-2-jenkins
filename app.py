import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import json
from PIL import Image

# filename = 'finalized_model.sav'
# model = pickle.load(open(filename, 'rb'))
logo = 'C5i_logo.jpg'

logo1 = Image.open(logo)

st.image(logo1, width=200) 

st.title('Sales Prediction Model')

TV = st.number_input(label = 'Television (in millions)',min_value=0.0,max_value=1000.0,step=0.1)

radio = st.number_input(label = 'Radio (in millions)',min_value=0.0,max_value=1000.0,step=0.1)

news = st.number_input(label = 'Newspaper (in millions)',min_value=0.0,max_value=1000.0,step=0.1)

#lis = np.array([sl,sw,pl,pw]).reshape(1,4)

inputs ={'TV':TV,'radio':radio,'news':news}

filename = 'multiple_reg.sav'
model = pickle.load(open(filename, 'rb'))

submit = st.button('submit')


if submit:
    #  out = model.predict(lis)
    #  st.success('The predicted output is {}'.format(out[0]))

    if TV==0 and radio==0 and news==0:
           st.warning('Give values to all the input fields')

    else:
            res =   np.array([TV,radio,news]).reshape(1,3)
            ans = model.predict(res)
            st.success('The Total Predicted Sales is {} millions'.format(round(ans[0][0],2)))