import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.cluster import DBSCAN
import sentence_transformers
import umap

def read_job_data(path=None):
    if path is None:
        path = '../Data/data_job_posts.csv'
    
    df = pd.read_csv(path)
    df.dropna(inplace=True, subset=['JobDescription', 'JobRequirment', 'RequiredQual'])
    df.reset_index(drop=True, inplace=True)
    return df

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    return text

def df_to_text(df):
    jobdf = df[['Title', 'JobRequirment', 'RequiredQual']].drop_duplicates()
    jobtxt = jobdf[['JobRequirment', 'RequiredQual']].values.tolist()
    jobtxt = [' '.join([clean_text(x) for x in job]) for job in jobtxt]
    return np.array(jobtxt)

def get_embeddings(text):
    model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text, show_progress_bar=True)
    return embeddings

def get_clusterer(embeddings, eps=0.5, min_samples=5):
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    clusterer.fit(embeddings)
    return clusterer

def get_umap(embeddings):
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.5)
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings
