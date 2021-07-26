# %%
from functools import cached_property
import json
import pickle
import base64
import re
from typing import Any, List
from urllib.request import urlopen

import streamlit as st
import pandas as pd
import numpy as np
from google.protobuf.symbol_database import Default
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import requests


# %%
class CompanyInfo:
    def __init__(self, ticker: str) -> None:
        res = requests.get(
            f'https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey=f33b3631d5140a4f1c87e7f2eafd8fdd')
        self.raw = res.json()[0]

    def __getattribute__(self, name: str) -> Any:
        if name != 'raw' and name in self.raw:
            return self.raw[name]
        return super().__getattribute__(name)

    @cached_property
    def df(self):
        return pd.DataFrame.from_dict([self.raw]).rename(columns={'0': 'Company Info'})


def generate_buttons(*labels: List[str], context=st):
    return [
        col.form_submit_button(label)
        for col, label in zip(
            context.beta_columns(len(labels)),
            labels
        )
    ]


def create_result(ticker: str, context=st):
    try:
        company_info = CompanyInfo(ticker)
    except:
        context.write(f'No company with {ticker} is found.')

    context.markdown(f"# {company_info.companyName}")
    col1, col2 = context.beta_columns([3, 6])
    col1.image(company_info.image)
    col1.write('''
    ## Company description:
    %s 
    ''' % company_info.description)

    return col1, col2

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
        cols = st.beta_columns(3)
        company_name = cols[0].text_input("Company Ticker", 'AAPL').upper()
        st.markdown('''<a href="https://drive.google.com/file/d/17etxeduBkckCdOH5WUElGEuULyFp_mIs">List of companies we provided</a>''',
                    unsafe_allow_html=True,)

        year = cols[1].text_input("Year (for prediction)", 2020, help='Our dataset ranges from year 2001 to 2020 (if the company existed on that year)')
        quarter = cols[2].slider('Quarter (for prediction)', min_value=1,
                            max_value=4, value=1, step=1)

        st.write('Search engine:')
        submitted, submitted_2, submbited_3 = generate_buttons(
            'Search Company Info',
            'Show financial statement',
            'Show figure dashboard',
        )

        st.write('Prediction engine:')
        submitted_4, submitted_5, submitted_6 = generate_buttons(
            'Bankruptcy prediction',
            'Earnings Call analysis',
            'Investment appraisal',
        )

    if submitted:
        try:
            company_info = CompanyInfo(company_name)
        except:
            st.write(f'No company with {company_name} is found.')

        col1, col2 = create_result(company_name)

        csv = company_info.df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{company_name}_info.csv">Download csv file for {company_info.companyName} info</a>'
        col1.markdown(href, unsafe_allow_html=True)

        col2.table(company_info.df.drop(columns=['companyName', 'description', 'image', 'defaultImage']).T)

    if submitted_2:

        col1, col2 = create_result(company_name)

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
                    col2.write(
                        f'No quarter info from {year} data with {company_name} is found.')
            except:
                col2.write(f'No {year} data with {company_name} is found.')
        except:
            col2.write(f'No company with {company_name} is found.')

        income = income.loc[(income['symbol'] == company_name) & (income['year'] == int(
            year)) & (income['period'] == int(quarter))]
        cashflow = cashflow.loc[(cashflow['symbol'] == company_name) & (cashflow['year'] == int(
            year)) & (cashflow['period'] == int(quarter))]

        col2.write(asset.drop(columns='Unnamed: 0').T)
        csv = asset.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{company_name}_asset.csv">Download csv file</a>'
        col2.markdown(href, unsafe_allow_html=True)

        col2.write(income.drop(columns='Unnamed: 0').T)
        csv = income.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{company_name}_income.csv">Download csv file</a>'
        col2.markdown(href, unsafe_allow_html=True)

        col2.write(cashflow.drop(columns='Unnamed: 0').T)
        csv = cashflow.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{company_name}_cashflow.csv">Download csv file</a>'
        col2.markdown(href, unsafe_allow_html=True)

    if submbited_3:

        col1, col2 = create_result(company_name)

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
        col1, col2 = create_result(company_name)

        pickle_in = open('model/forest.pkl', 'rb')
        classifier = pickle.load(pickle_in)

        df = pd.read_csv('data/us_bankruptcy_noinf.csv')
        try:
            df = df.loc[df['symbol'] == company_name]
            try:
                df = df.loc[df['year'] == int(year)]
                try:
                    df = df.loc[df['period'] == int(quarter)]
                    col2.write(df.iloc[:, 1:].T)
                except:
                    col2.write(
                        f'No quarter info from {year} data with {company_name} is found.')
            except:
                col2.write(f'No {year} data with {company_name} is found.')
        except:
            col2.write(f'No company with {company_name} is found.')

        prediction = classifier.predict(df.iloc[:, 5:])
        if prediction == 0:
            col2.write(
                f'At the moment, {company_name} is unlikely to be bankrupt according to prediction.')
        else:
            col2.write(
                f'Warning: {company_name} is likely to be bankrupt within a year according to prediction.')

    # def word_preprocess(n):
    #     # --- Remove separaters
    #     n = re.sub(r'\n', '', n)
    #     n = re.sub(r'--', '', n)
    #     # --- Remove stopword
    #     stop_words = set(stopwords.words('english')+list(punctuation))
    #     words_token = [w for w in word_tokenize(
    #         n) if not w.lower() in stop_words]
    #     filtered_sentence = [i for i in words_token if i not in stop_words]
    #     # --- Lemmatize
    #     lemmatizer = WordNetLemmatizer()
    #     clean_list = ",".join([lemmatizer.lemmatize(i)
    #                            for i in filtered_sentence])
    #     return clean_list

    if submitted_5:

        col1, col2 = create_result(company_name)

        col2.markdown(
            '''
            ## Sentiment analysis (with TextBlob)
            The polarity score is a float within the range [-1.0, 1.0].
            The subjectivity is a float within the range [0.0, 1.0] where
            0.0 is very objective and 1.0 is very subjective.'''
        )

        polarity_arr = []
        subjectivity_arr = []
        for i in range(1, 5):
            try:
                url = (
                    f'https://financialmodelingprep.com/api/v3/earning_call_transcript/{company_name}?quarter={i}&year={year}&apikey=f33b3631d5140a4f1c87e7f2eafd8fdd')
                response = urlopen(url)
            except:
                st.write(
                    f'No earnings call with {company_name} on year {year} quarter {i} is found.')
            data = response.read().decode("utf-8")
            polarity_arr.append(TextBlob(data).sentiment.polarity)
            subjectivity_arr.append(TextBlob(data).sentiment.subjectivity)

        col2.bar_chart(pd.DataFrame(
            {'polarity': polarity_arr}, index=['Q1', 'Q2', 'Q3', 'Q4']))
        col2.bar_chart(pd.DataFrame(
            {'subjectivity': subjectivity_arr}, index=['Q1', 'Q2', 'Q3', 'Q4']))

        # pickle_in = open('model/tree.pkl', 'rb')
        # classifier = pickle.load(pickle_in)

        # cv = CountVectorizer()
        # data_x = cv.fit_transform(list(data2))
        # criteria = ['threeYRevenueGrowthPerShare', 'fiveYRevenueGrowthPerShare', 'tenYRevenueGrowthPerShare',
        #             'threeYDividendperShareGrowthPerShare', 'fiveYDividendperShareGrowthPerShare', 'tenYDividendperShareGrowthPerShare']
        # for i in criteria:
        #     pickle_in = open(f'model/{i}.pkl', 'rb')
        #     classifier = pickle.load(pickle_in)
        #     prediction = classifier.predict(data_x)
        #     st.write(f'{i}, {prediction}')


# %%
if __name__ == '__main__':
    webpage()

# %%
