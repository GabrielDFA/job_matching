import streamlit as st

def upload_cv_view():
    st.title('CV Job Matching Application')
    st.write("Upload your CV in PDF format to find matching jobs.")
    cv_file = st.file_uploader("Upload CV", type=["pdf"])
    
    if cv_file is not None:
        st.success("CV uploaded successfully!")
        return cv_file
    return None

def display_matching_jobs(matching_jobs):
    if not matching_jobs.empty:
        matching_companies = matching_jobs[['company', 'job_title', 'location', 'category', 'subcategory', 'role', 'type', 'salary', 'listingDate']]
        st.write(f"Matching Companies Found: {len(matching_companies['company'].unique())}")
        st.dataframe(matching_companies.drop_duplicates(subset=['company']))
    else:
        st.write("No matching jobs found.")
