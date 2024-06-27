from models.models import JobDataset, CVProcessor

class JobMatcherController:
    def __init__(self, dataset_path):
        self.job_dataset = JobDataset(dataset_path)
    
    def match_cv(self, cv_file):
        cv_text = CVProcessor.extract_text_from_pdf(cv_file)
        matching_jobs = CVProcessor.match_skills(cv_text, self.job_dataset.get_jobs())
        return matching_jobs
