# %%
import streamlit as st
import pandas as pd
import numpy as np
import json
from urllib.request import urlopen
import pickle
import sklearn
import base64

# %%


# @st.cache(suppress_st_warning=True)
def webpage():

    st.set_page_config(page_title='Monitus', layout="wide",
                       initial_sidebar_state='collapsed')

    dum1, dum2, dum3 = st.beta_columns([1, 6, 1])
    with dum1:
        st.write("")

    with dum2:
        st.image('image/logo_long.png', width=None)

    with dum3:
        st.write("")

    # display the front end aspect
    html_temp = """
    <h3 style ="color:black;text-align:center;">Financial Statement Analysis using machine learning tools.</h3>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")

    with st.form('Company Information'):

        st.write('Choose a company for prediction:')
        company_name = st.text_input("Company Name", 'AAPL')
        st.markdown('''<a href="https://drive.google.com/file/d/17etxeduBkckCdOH5WUElGEuULyFp_mIs">List of companies we provided</a>''',
                    unsafe_allow_html=True,)
        year = st.text_input("Year (for prediction)", 2020)
        st.markdown(
            'Our dataset ranges from year 2001 to 2020 (if the company existed on that year)')
        quarter = st.slider('Quarter (for prediction)', min_value=1,
                            max_value=4, value=1, step=1)
        st.write('Search engine:')
        col1, col2, col3 = st.beta_columns(3)
        submitted = col1.form_submit_button("Search Company Info")
        submitted_2 = col2.form_submit_button("Show financial statement")
        submbited_3 = col3.form_submit_button("Show figure dashboard")
        st.write('Prediction engine:')
        col1, col2, col3 = st.beta_columns(3)
        submitted_4 = col1.form_submit_button('Bankruptcy prediction')
        submitted_5 = col2.form_submit_button('Investment appraisal')
        submitted_6 = col3.form_submit_button('Earnings Call analysis')

    if submitted:
        try:
            url_comp = urlopen(
                f'https://financialmodelingprep.com/api/v3/profile/{company_name}?apikey=f33b3631d5140a4f1c87e7f2eafd8fdd')
        except:
            st.write(f'No company with {company_name} is found.')
        data = json.loads(url_comp.read().decode('utf-8'))
        df = pd.DataFrame.from_dict(data).rename(columns={'0': 'Company Info'})
        col1, col2 = st.beta_columns(2)

        temp = data[0]['companyName']
        col1.markdown(f"**{temp}**")
        col1.image(data[0]['image'])
        col1.dataframe(
            df.drop(columns=['companyName', 'description', 'image', 'defaultImage']).T)
        col2.write('Company description:')
        col2.write(data[0]['description'])

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{company_name}_info.csv">Download csv file for {temp} info</a>'
        st.markdown(href, unsafe_allow_html=True)

    if submitted_2:

        try:
            url_comp = urlopen(
                f'https://financialmodelingprep.com/api/v3/profile/{company_name}?apikey=f33b3631d5140a4f1c87e7f2eafd8fdd')
        except:
            st.write(f'No company with {company_name} is found.')
        data = json.loads(url_comp.read().decode('utf-8'))
        df = pd.DataFrame.from_dict(data).rename(columns={'0': 'Company Info'})
        col1, col2 = st.beta_columns(2)

        temp = data[0]['companyName']
        col1.markdown(f"**{temp}**")
        col1.image(data[0]['image'])

        asset = pd.read_csv('data/asset_df.csv')
        income = pd.read_csv('data/income_df.csv')
        cashflow = pd.read_csv('data/cashflow_df.csv')

        # --- Validation of data
        try:
            asset = asset.loc[asset['symbol'] == company_name]
            try:
                asset = asset.loc[asset['year'] == int(year)]
                try:
                    asset = asset.loc[asset['period'] == int(quarter)]
                except:
                    st.write(
                        f'No quarter info from {year} data with {company_name} is found.')
            except:
                st.write(f'No {year} data with {company_name} is found.')
        except:
            st.write(f'No company with {company_name} is found.')

        income = income.loc[(income['symbol'] == company_name) & (income['year'] == int(
            year)) & (income['period'] == int(quarter))]
        cashflow = cashflow.loc[(cashflow['symbol'] == company_name) & (cashflow['year'] == int(
            year)) & (cashflow['period'] == int(quarter))]

        st.write(asset.drop(columns='Unnamed: 0').T)
        csv = asset.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{company_name}_asset.csv">Download csv file</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.write(income.drop(columns='Unnamed: 0').T)
        csv = income.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{company_name}_income.csv">Download csv file</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.write(cashflow.drop(columns='Unnamed: 0').T)
        csv = cashflow.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{company_name}_cashflow.csv">Download csv file</a>'
        st.markdown(href, unsafe_allow_html=True)

    if submbited_3:

        try:
            url_comp = urlopen(
                f'https://financialmodelingprep.com/api/v3/profile/{company_name}?apikey=f33b3631d5140a4f1c87e7f2eafd8fdd')
        except:
            st.write(f'No company with {company_name} is found.')
        data = json.loads(url_comp.read().decode('utf-8'))
        df = pd.DataFrame.from_dict(data).rename(columns={'0': 'Company Info'})
        col1, col2 = st.beta_columns(2)

        temp = data[0]['companyName']
        col1.markdown(f"**{temp}**")
        col1.image(data[0]['image'])

        ratio = pd.read_csv('data/ratio_df.csv')
        growth = pd.read_csv('data/growth_df.csv')

        # --- Validation of data
        try:
            ratio = ratio.loc[ratio['symbol'] == company_name]
            try:
                ratio = ratio.loc[ratio['year'] == int(year)]
                try:
                    ratio = ratio.loc[ratio['period'] == int(quarter)]
                except:
                    st.write(
                        f'No quarter info from {year} data with {company_name} is found.')
            except:
                st.write(f'No {year} data with {company_name} is found.')
        except:
            st.write(f'No company with {company_name} is found.')

        growth = growth.loc[(growth['symbol'] == company_name) & (growth['year'] == int(
            year)) & (growth['period'] == f'Q{quarter}')]

        st.write(ratio.drop(columns='Unnamed: 0').T)

        csv = ratio.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{company_name}_ratio.csv">Download csv file</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.bar_chart(data=ratio.drop(
            columns=['Unnamed: 0', 'quarter', 'year', 'period', 'symbol', 'date', 'daysOfSalesOutstanding', 'daysOfInventoryOutstanding', 'operatingCycle', 'daysOfPayablesOutstanding', 'cashConversionCycle']).T)
        st.bar_chart(data=ratio[['daysOfSalesOutstanding', 'daysOfInventoryOutstanding',
                                 'operatingCycle', 'daysOfPayablesOutstanding', 'cashConversionCycle']].T)

        st.write(growth.drop(columns='Unnamed: 0').T)

        csv = growth.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{company_name}_growth.csv">Download csv file</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.bar_chart(data=growth.drop(
            columns=['Unnamed: 0', 'quarter', 'year', 'period', 'symbol', 'date']).T)

    if submitted_4:

        try:
            url_comp = urlopen(
                f'https://financialmodelingprep.com/api/v3/profile/{company_name}?apikey=f33b3631d5140a4f1c87e7f2eafd8fdd')
        except:
            st.write(f'No company with {company_name} is found.')
        data = json.loads(url_comp.read().decode('utf-8'))
        df = pd.DataFrame.from_dict(data).rename(columns={'0': 'Company Info'})
        col1, col2 = st.beta_columns(2)

        temp = data[0]['companyName']
        col1.markdown(f"**{temp}**")
        col1.image(data[0]['image'])

        pickle_in = open('classifier.pkl', 'rb')
        classifier = pickle.load(pickle_in)

        df = pd.read_csv('data/us_bankruptcy_noinf.csv')

        bankruptcy(company_name, year, quarter, classifier)

    def bankruptcy(company_name, year, quarter, model):
        prediction = model.predict([])


# %%
if __name__ == '__main__':
    webpage()

# %%
