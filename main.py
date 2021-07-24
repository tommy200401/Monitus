# %%
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from urllib.request import urlopen
import pickle
import sklearn

# %%


# @st.cache(suppress_st_warning=True)
def webpage():

    # display the front end aspect
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.image('image/logo_long.png')
    st.write('You can search the company information and prediction using this tool.')

    with st.sidebar.form('Company Information'):

        st.write('Choose a company for prediction:')
        company_name = st.text_input("Company Name", 'AAPL')
        st.markdown('''<a href="https://www.google.com/">List of companies we provided</a>''',
                    unsafe_allow_html=True,)
        year = st.text_input("Year", 2020)
        quarter = st.slider('Quarter', min_value=1,
                            max_value=4, value=1, step=1)
        submitted = st.form_submit_button("Search Company Info")
        submitted_2 = st.form_submit_button("Predict bankruptcy")

    if submitted:
        # with open('data/profile.txt', 'rb') as f:
        #     temp = json.loads(f.read())
        url_comp = urlopen(
            f'https://financialmodelingprep.com/api/v3/profile/{company_name}?apikey=f33b3631d5140a4f1c87e7f2eafd8fdd')
        data = json.loads(url_comp.read().decode('utf-8'))

        df = pd.DataFrame.from_dict(data).rename(columns={'0': 'Company Info'})
        st.write(data[0]['companyName'])
        st.image(data[0]['image'])
        st.dataframe(
            df.drop(columns=['companyName', 'description', 'image', 'defaultImage']).T)
        st.write('Company description:')
        st.write(data[0]['description'])

    if submitted_2:
        # load prediction model
        with open('model/tree.pkl', 'rb') as pickle_in:
            classifier = pickle.load(pickle_in)
            pickle_in.close()
        us = pd.read_csv('data/us_bankruptcy.csv')

        st.write(us.iloc[0])


# %%
if __name__ == '__main__':
    webpage()

# %%
