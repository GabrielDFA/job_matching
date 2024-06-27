import streamlit as st
from views.views import upload_cv_view, display_matching_jobs
from controllers.controllers import JobMatcherController

def main():
    dataset_path = 'data/jobstreet_all_job_dataset.csv'
    controller = JobMatcherController(dataset_path)
    
    st.sidebar.title('Navigation')
    options = st.sidebar.radio('Go to', ['Upload CV'])

    if options == 'Upload CV':
        cv_file = upload_cv_view()
    
        if cv_file:
            matching_jobs = controller.match_cv(cv_file)
            display_matching_jobs(matching_jobs)

if __name__ == "__main__":
    main()
