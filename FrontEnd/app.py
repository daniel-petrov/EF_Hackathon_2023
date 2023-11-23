import streamlit as st
import pandas as pd
import numpy as np

st.title('Job Postings Clustering')

def load_data():
    df = pd.read_csv('../Data/data_job_posts.csv')
    df.dropna(inplace=True, subset=['JobDescription', 'JobRequirment', 'RequiredQual'])
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data()
st.subheader('Data Sample')
st.write(df.head())