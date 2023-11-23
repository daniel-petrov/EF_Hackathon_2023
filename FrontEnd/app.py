import streamlit as st
import pandas as pd
import re
from pdfminer.high_level import extract_text
import sentence_transformers
import numpy as np

@st.cache_data
def load_data():
    """ 
    Load job postings dataset
    """
    df = pd.read_csv('../Data/data_job_posts.csv')
    df.dropna(inplace=True, subset=['JobDescription', 'JobRequirment', 'RequiredQual'])
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def load_embeddings():
    """ 
    Load pre-trained clustering model
    """
    embeddings = pd.read_pickle('../Data/embeddings.pkl')
    
    return embeddings

def parse_cv(cv_file):
    """ 
    Parse CV and return embeddings
    """
    raw_text = extract_text(cv_file)
    formatted_text = re.sub(r'[^A-Za-z0-9]+', ' ', raw_text)
    model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(formatted_text, show_progress_bar=False)
    
    return embeddings


def find_jobs(jobs, embedded_cv, embedded_jobs, n_jobs=10):
    """ 
    Find jobs based on CV embeddings
    """
    distances = np.linalg.norm(embedded_jobs - embedded_cv, axis=1)
    closest = np.argsort(distances)[:n_jobs]
    
    return jobs.iloc[closest][['Title', 'JobDescription', 'JobRequirment', 'RequiredQual']]



def show_homepage():
    """ 
    Show main streamlit page
    """
    st.title('Job Postings Clustering')
    st.write('This web app is a demo for clustering job postings. It uses a dataset of job postings from different fields and clusters them into 10 categories.')
    st.subheader('Project info')
    st.write('''
    - The dataset was taken from [Kaggle](https://www.kaggle.com/madhab/jobposts).
    - The clustering model was trained using [Sentence Transformers](https://www.sbert.net/).
    - The web app was built using [Streamlit](https://streamlit.io/).
    - The source code can be found on [GitHub](https://github.com/daniel-petrov/EF_Hackathon_2023)
             ''')

    uploaded_cv = st.file_uploader('Upload your CV as a pdf 👇', type=["pdf"], accept_multiple_files=False)
    if uploaded_cv:
        if uploaded_cv.type == 'application/pdf':
            parsed_cv = parse_cv(uploaded_cv)
            st.success('CV uploaded successfully!')
        else:
            st.warning('Please upload a PDF file')

    if 'parsed_cv' in locals():
        if st.button('Find me a job!'):
            # Display top n jobs based off uploaded CV
            jobs = load_data()
            embedded_jobs = load_embeddings()

            top_jobs = find_jobs(jobs, parsed_cv, embedded_jobs, n_jobs=10)
            st.write(top_jobs)



show_homepage()