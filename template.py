import os 
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "job_classification_project"

list_of_files = [
    ".github/workflows/.gitkeep",

    f"{project_name}/data/.gitkeep",

    f"{project_name}/src/__init__.py",
    f"{project_name}/src/config.py",
    f"{project_name}/src/load_data.py",
    f"{project_name}/src/preprocess.py",
    f"{project_name}/src/eda.py",
    f"{project_name}/src/model.py",
    f"{project_name}/src/evaluate.py",

    f"{project_name}/main.py",
    f"{project_name}/requirements.txt",
    f"{project_name}/README.md",
    f"{project_name}/.gitignore",
    f"{project_name}/research/eda.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
