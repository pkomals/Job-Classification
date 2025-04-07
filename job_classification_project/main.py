from src.load_data import load_job_data, load_resume_data

if __name__ == "__main__":
    # Replace with your actual GitHub RAW URLs
    job_data_url = "https://raw.githubusercontent.com/pkomals/Data/refs/heads/main/Linkedin%20job%20listings%20information.csv"
    resume_data_url = "https://raw.githubusercontent.com/pkomals/Data/refs/heads/main/UpdatedResumeDataSet.csv"

    job_df = load_job_data(job_data_url)
    resume_df = load_resume_data(resume_data_url)

    print(job_df.head())
    print(resume_df.head())
