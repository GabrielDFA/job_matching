import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from PyPDF2 import PdfReader

# Indonesian stop words
INDONESIAN_STOP_WORDS = [
    'yang', 'untuk', 'pada', 'dengan', 'ini', 'itu', 'dan', 'di', 'ke', 'dari', 'adalah', 'sebagai', 'juga', 
    'dalam', 'tidak', 'akan', 'atau', 'sudah', 'seperti', 'karena', 'oleh', 'dapat', 'bagi', 'mereka', 'namun', 
    'setelah', 'lebih', 'harus', 'tersebut', 'kami', 'diri', 'sendiri', 'saya', 'kita', 'ada', 'saat', 'sedang', 
    'hanya', 'itu', 'ini', 'saja', 'lain', 'jadi', 'bukan', 'lagi', 'maka', 'pun', 'apa', 'baik', 'agar', 'tanpa', 
    'banyak', 'sebuah', 'bawah', 'jika', 'semua', 'yaitu', 'sesuatu', 'kapan', 'maupun', 'kemudian', 'sebelum', 
    'kembali', 'menjadi', 'kepada', 'lebih', 'masih', 'antara', 'oleh', 'setiap', 'orang', 'terhadap', 'jika', 
    'belum', 'sesudah', 'setiap', 'kami', 'para', 'membuat', 'tidak', 'ada', 'cara', 'menggunakan', 'seperti', 
    'atas', 'sendiri', 'agar', 'seperti', 'dari', 'dengan', 'tetapi', 'meski', 'namun', 'kalau', 'melakukan', 
    'sehingga', 'apabila', 'berada', 'bahwa', 'begitu', 'bisa', 'bukan', 'tidak', 'universitas', 'terakhir', 'belakang', 'akhir'
]

# Combined stop words
COMBINED_STOP_WORDS = list(set(ENGLISH_STOP_WORDS).union(INDONESIAN_STOP_WORDS).union({
    'experience', 'skills', 'responsibilities', 'requirements', 
    'position', 'role', 'job', 'title', 'company', 'location', 
    'salary', 'work', 'include', 'required', 'preferred', 
    'qualifications', 'applicant', 'applications'
}))

class JobDataset:
    def __init__(self, part_files):
        self.dataset = self.load_and_combine(part_files)
    
    def load_and_combine(self, part_files):
        df_list = [pd.read_csv(file) for file in part_files]
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df = combined_df.dropna(subset=['role'])  # Drop rows with NaN in 'role' column
        return combined_df

    def get_jobs(self):
        return self.dataset

class CVProcessor:
    CUSTOM_STOP_WORDS = COMBINED_STOP_WORDS

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
        job_titles = job_dataset['role'].tolist()
        job_titles = [str(title) for title in job_titles if pd.notnull(title)]  # Ensure no NaN values
        texts = [cv_text] + job_titles
        
        # TF-IDF Vectorization with custom stop words and min_df
        vectorizer = TfidfVectorizer(stop_words=CVProcessor.CUSTOM_STOP_WORDS, min_df=2)
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
        
        return job_matches.sort_values(by='similarity', ascending=False), vectorizer, cv_vector

class JobMatcherController:
    def __init__(self, part_files):
        self.job_dataset = JobDataset(part_files)
    
    def match_cv(self, cv_file):
        cv_text = CVProcessor.extract_text_from_pdf(cv_file)
        matching_jobs, vectorizer, cv_vector = CVProcessor.match_skills(cv_text, self.job_dataset.get_jobs())
        return matching_jobs, vectorizer, cv_vector, cv_text

def paginate(dataframe, page_size=10):
    total_pages = min((len(dataframe) - 1) // page_size + 1, 10)  # Limit to 10 pages
    page_number = st.sidebar.number_input('Page Number', min_value=1, max_value=total_pages, value=1)
    start_index = (page_number - 1) * page_size
    end_index = start_index + page_size
    return dataframe.iloc[start_index:end_index], total_pages, page_number

def upload_cv_view():
    st.title('CV Job Matching Application')
    st.write("Upload your CV in PDF format to find matching jobs.")
    cv_file = st.file_uploader("Upload CV", type=["pdf"])
    
    if cv_file is not None:
        st.success("CV uploaded successfully!")
        return cv_file
    return None

def display_matching_jobs(matching_jobs, vectorizer, cv_vector, cv_text, page_size=10):
    if not matching_jobs.empty:
        paginated_jobs, total_pages, page_number = paginate(matching_jobs, page_size)
        
        st.write(f"Page {page_number} of {total_pages}")
        matching_companies = paginated_jobs[['company', 'job_title', 'location', 'category', 'subcategory', 'role', 'type', 'salary', 'listingDate']]
        st.write(f"Matching Companies Found: {len(matching_companies['company'].unique())}")
        st.dataframe(matching_companies.drop_duplicates(subset=['company']))

        for index, row in paginated_jobs.iterrows():
            st.subheader(f"{row['job_title']} at {row['company']}")
            st.write(f"Location: {row['location']}")
            st.write(f"Category: {row['category']} | Subcategory: {row['subcategory']}")
            st.write(f"Role: {row['role']} | Type: {row['type']}")
            st.write(f"Salary: {row['salary']} | Listing Date: {row['listingDate']}")
            st.write(f"Similarity Score: {row['similarity']:.2f}")
            explanation = get_explanation(cv_text, row['job_title'], vectorizer, cv_vector)
            st.write(f"Why this job matches your CV:\n{explanation}")
            st.write("---")
    else:
        st.write("No matching jobs found.")

def get_explanation(cv_text, job_title, vectorizer, cv_vector):
    job_title_vector = vectorizer.transform([job_title])
    feature_names = vectorizer.get_feature_names_out()
    cv_feature_indices = cv_vector.nonzero()[1]
    job_feature_indices = job_title_vector.nonzero()[1]
    
    cv_features = [feature_names[i] for i in cv_feature_indices]
    job_features = [feature_names[i] for i in job_feature_indices]
    
    common_features = set(cv_features).intersection(set(job_features))
    explanation = ", ".join(common_features)
    
    if not explanation:
        explanation = "The job title and your CV have general matching keywords."
    
    return explanation

def main():
    part_files = ['dataset_part1.csv', 'dataset_part2.csv']  
    controller = JobMatcherController(part_files)
    
    st.sidebar.title('Navigation')
    options = st.sidebar.radio('Go to', ['Upload CV'])

    if options == 'Upload CV':
        cv_file = upload_cv_view()
    
        if cv_file:
            matching_jobs, vectorizer, cv_vector, cv_text = controller.match_cv(cv_file)
            display_matching_jobs(matching_jobs, vectorizer, cv_vector, cv_text)

if __name__ == "__main__":
    main()
