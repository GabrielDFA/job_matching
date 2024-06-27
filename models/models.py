from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from PyPDF2 import PdfReader
import re

class JobDataset:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
    
    def get_jobs(self):
        return self.dataset

class CVProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_file):
        pdf_reader = PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def match_skills(cv_text, job_dataset):
        # Combine CV text and job titles for TF-IDF vectorization
        job_titles = job_dataset['job_title'].tolist()
        texts = [cv_text] + job_titles
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Extract the CV vector and job title vectors
        cv_vector = tfidf_matrix[0:1]
        job_vectors = tfidf_matrix[1:]
        
        # Calculate cosine similarities
        similarities = cosine_similarity(cv_vector, job_vectors).flatten()
        
        # Add similarities to the dataset
        job_dataset['similarity'] = similarities
        
        # Filter jobs based on a similarity threshold
        job_matches = job_dataset[job_dataset['similarity'] > 0.1]
        
        return job_matches.sort_values(by='similarity', ascending=False)
