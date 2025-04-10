import os
import shutil

def create_directory_structure():
    # Define the directory structure
    structure = {
        'data': ['job_descriptions.csv'],
        'src': ['data_preprocessing.py', 'job_classifier.py', 'job_recommender.py', 'main.py', 'requirements.txt'],
        'artifacts': ['processed_data.csv'],
        'models': ['job_classifier.joblib']
    }
    
    # Create directories and files
    for directory, files in structure.items():
        # Create directory
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
        
        # Create empty files in the directory
        for file in files:
            file_path = os.path.join(directory, file)
            with open(file_path, 'w') as f:
                f.write('')
            print(f"Created file: {file_path}")

if __name__ == "__main__":
    print("Creating project directory structure...")
    create_directory_structure()
    print("\nProject structure created successfully!") 