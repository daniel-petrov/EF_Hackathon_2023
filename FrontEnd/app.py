import streamlit as st
import pandas as pd
import re
from pdfminer.high_level import extract_text
import sentence_transformers
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

@st.cache_data
def load_data():
    """ 
    Load job postings dataset
    """
    df = pd.read_pickle('../Data/data_job_posts.pkl')
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

def get_text_from_pdf(cv_file):
    """ 
    Extract text from CV
    """
    raw_text = extract_text(cv_file)
    formatted_text = re.sub(r'[^A-Za-z0-9]+', ' ', raw_text)
    
    return formatted_text

def parse_cv(cv_file):
    """ 
    Parse CV and return embeddings
    """
    raw_text = extract_text(cv_file)
    formatted_text = re.sub(r'[^A-Za-z0-9]+', ' ', raw_text)
    model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(formatted_text, show_progress_bar=False)
    
    return embeddings
@st.cache_data
def find_jobs(jobs, embedded_cv, embedded_jobs, n_jobs=10):
    """ 
    Find jobs based on CV embeddings
    """
    distances = np.linalg.norm(embedded_jobs - embedded_cv, axis=1)
    closest = np.argsort(distances)[:n_jobs]
    
    return jobs.iloc[closest][['Title', 'JobDescription', 'JobRequirment', 'RequiredQual']]

def generate_cover_letter(cv, job_desc):
    """ 
    Generate cover letter
    """
    load_dotenv()
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Create a cover letter based on the following CV and job description."},
            {"role": "user", "content": f"CV:\n{cv}\n\nJob Description:\n{job_desc}"}
        ]
    )

    cover_letter = response.choices[0].message.content
    return cover_letter

# Initialize session state variables
def initialize_state():
    if 'top_job_titles_to_display' not in st.session_state:
        st.session_state['top_job_titles_to_display'] = None
    if 'top_jobs' not in st.session_state:
        st.session_state['top_jobs'] = None





def show_homepage():
    """ 
    Show main streamlit page
    """
    
    # Initialize session state variables
    initialize_state()

    st.title('JobSeeker AI')
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
        max_jobs = 20
        num_jobs = st.slider('How many jobs do you want to see?', 1, max_jobs, 5)
        
        if st.button(f'Suggest me {num_jobs} jobs!'):
            # Display top n jobs based off uploaded CV
            jobs = load_data()
            embedded_jobs = load_embeddings()
            
            top_jobs = find_jobs(jobs, parsed_cv, embedded_jobs, n_jobs=max_jobs)
            top_job_titles = top_jobs.iloc[:, 0].to_frame()

            # Store the DataFrame in session state
            st.session_state['top_job_titles_to_display'] = top_job_titles.head(num_jobs)

        # Display the DataFrame from session state
        if st.session_state['top_job_titles_to_display'] is not None:
            st.dataframe(st.session_state['top_job_titles_to_display'], hide_index=True)

        # Generate a select box only if the DataFrame is available in session state
        if st.session_state['top_job_titles_to_display'] is not None:
            option = st.selectbox('Generate cover letter for:', st.session_state['top_job_titles_to_display'])
            if st.button(f'Generate a cover letter for {option}'):
                jobs = load_data()
                embedded_jobs = load_embeddings()
                top_jobs = find_jobs(jobs, parsed_cv, embedded_jobs, n_jobs=max_jobs)
                
                job_description = top_jobs[top_jobs['Title'] == option]['JobDescription'].values[0]
                cv_text = get_text_from_pdf(uploaded_cv)
                #st.write(job_description)
                #st.write(cv_text)

                cover_letter = generate_cover_letter(cv_text, job_description)
                st.write(cover_letter)

            



show_homepage()