import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import os

def load_data():
    """Load and return the job descriptions dataset"""
    try:
        df = pd.read_csv(r'C:\Projects\Job Recommendation\data\job_descriptions.csv')
        print("Data loaded successfully!")
        return df
    except FileNotFoundError:
        print("Error: Could not find job descriptions data file.")
        return None

def basic_info(df):
    """Print basic information about the dataset"""
    print("\nBasic Information:")
    print("=" * 50)
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())

def analyze_job_titles(df):
    """Analyze job titles"""
    print("\nJob Title Analysis:")
    print("=" * 50)
    
    # Count unique job titles
    unique_titles = df['Job Title'].nunique()
    print(f"Number of unique job titles: {unique_titles}")
    
    # Top 10 most common job titles
    print("\nTop 10 Most Common Job Titles:")
    title_counts = df['Job Title'].value_counts().head(10)
    print(title_counts)
    
    # Plot job title distribution
    plt.figure(figsize=(12, 6))
    title_counts.plot(kind='bar')
    plt.title('Top 10 Most Common Job Titles')
    plt.xlabel('Job Title')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('job_title_distribution.png')
    plt.close()

def analyze_descriptions(df):
    """Analyze job descriptions"""
    print("\nJob Description Analysis:")
    print("=" * 50)
    
    # Calculate description lengths
    df['description_length'] = df['Job Description'].str.len()
    
    # Basic statistics
    print("\nDescription Length Statistics:")
    print(df['description_length'].describe())
    
    # Plot description length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['description_length'], bins=50)
    plt.title('Distribution of Job Description Lengths')
    plt.xlabel('Description Length (characters)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('description_length_distribution.png')
    plt.close()

def analyze_skills(df):
    """Analyze required skills"""
    print("\nSkills Analysis:")
    print("=" * 50)
    
    # Extract skills from descriptions
    skills_list = []
    for desc in df['Job Description']:
        # Look for common skill indicators
        skills = re.findall(r'(?i)(python|java|c\+\+|sql|machine learning|data analysis|aws|azure|docker|kubernetes|react|angular|node\.js|tensorflow|pytorch)', str(desc))
        skills_list.extend(skills)
    
    # Count skill occurrences
    skill_counts = Counter(skills_list)
    
    # Top 10 most required skills
    print("\nTop 10 Most Required Skills:")
    for skill, count in skill_counts.most_common(10):
        print(f"{skill}: {count}")
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(skill_counts)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Required Skills Word Cloud')
    plt.tight_layout()
    plt.savefig('skills_wordcloud.png')
    plt.close()

def analyze_experience(df):
    """Analyze experience requirements"""
    print("\nExperience Requirements Analysis:")
    print("=" * 50)
    
    # Extract experience requirements
    experience_patterns = [
        r'(\d+)\+?\s*(?:years|yrs|yr)',
        r'(\d+)\s*-\s*(\d+)\s*(?:years|yrs|yr)'
    ]
    
    experience_counts = Counter()
    for desc in df['Job Description']:
        for pattern in experience_patterns:
            matches = re.findall(pattern, str(desc).lower())
            if matches:
                if isinstance(matches[0], tuple):
                    # For ranges like "2-5 years"
                    exp_range = f"{matches[0][0]}-{matches[0][1]} years"
                    experience_counts[exp_range] += 1
                else:
                    # For single numbers like "5+ years"
                    exp = f"{matches[0]}+ years"
                    experience_counts[exp] += 1
    
    print("\nExperience Requirements Distribution:")
    for exp, count in experience_counts.most_common():
        print(f"{exp}: {count}")

def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create output directory for plots
    os.makedirs('eda_output', exist_ok=True)
    
    # Perform EDA
    basic_info(df)
    analyze_job_titles(df)
    analyze_descriptions(df)
    analyze_skills(df)
    analyze_experience(df)
    
    print("\nEDA completed! Check the 'eda_output' directory for visualizations.")

if __name__ == "__main__":
    main() 