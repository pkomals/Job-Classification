import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
import os
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from sklearn.preprocessing import LabelEncoder

class JobRecommender:
    def __init__(self):
        # Initialize vectorizers
        self.skills_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.desc_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Load job listings data
        try:
            self.job_listings_df = pd.read_csv(r'C:\Projects\Job Recommendation\data\synthetic_job_data.csv')
            print("Job listings data loaded successfully!")
            
            # Fit vectorizers on the loaded data
            self._fit_vectorizers()
            
        except FileNotFoundError:
            print("Error: Could not find job listings data file.")
            raise
    
    def _fit_vectorizers(self):
        """Fit vectorizers on the job listings data"""
        # Fit skills vectorizer
        self.skills_vectorizer.fit(self.job_listings_df['skills'].fillna(''))
        
        # Fit description vectorizer
        self.desc_vectorizer.fit(self.job_listings_df['Job Description'].fillna(''))
    
    def preprocess_text(self, text):
        """Preprocess text for vectorization"""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def _calculate_experience_match(self, user_exp, job_exp):
        """Calculate experience match score"""
        exp_mapping = {
            "0-2 years": 1,
            "2-5 years": 2,
            "5-8 years": 3,
            "8+ years": 4
        }
        
        user_level = exp_mapping.get(user_exp, 0)
        job_level = exp_mapping.get(job_exp, 0)
        
        if user_level >= job_level:
            return 1.0
        elif user_level == job_level - 1:
            return 0.7
        else:
            return 0.3
    
    def _calculate_education_match(self, user_edu, job_edu):
        """Calculate education match score"""
        edu_mapping = {
            "Bachelor's Degree": 1,
            "Bachelor's Degree or equivalent experience": 1,
            "Master's Degree": 2,
            "Master's Degree or equivalent experience": 2,
            "PhD": 3
        }
        
        user_level = edu_mapping.get(user_edu, 0)
        job_level = edu_mapping.get(job_edu, 0)
        
        if user_level >= job_level:
            return 1.0
        elif user_level == job_level - 1:
            return 0.7
        else:
            return 0.3
    
    def get_job_recommendations(self, user_profile, top_n=5):
        """
        Get job recommendations based on user profile
        
        Args:
            user_profile (dict): Dictionary containing user information
                {
                    'skills': str,  # Comma-separated list of skills
                    'education': str,  # Education level
                    'experience': str,  # Experience range
                    'preferred_industry': str  # Optional
                }
            top_n (int): Number of top recommendations to return
            
        Returns:
            list: List of dictionaries containing recommendations and scores
                [{
                    'job_title': str,
                    'company': str,
                    'job_description': str,
                    'required_skills': str,
                    'matching_score': float,
                    'skill_match_score': float,
                    'education_match_score': float,
                    'experience_match_score': float,
                    'category_match_score': float
                }]
        """
        # Preprocess user skills
        user_skills = self.preprocess_text(user_profile['skills'])
        
        # Vectorize user skills
        user_skills_vector = self.skills_vectorizer.transform([user_skills])
        
        recommendations = []
        
        for _, job in self.job_listings_df.iterrows():
            # Preprocess job information
            job_skills = self.preprocess_text(job['skills'])
            job_desc = self.preprocess_text(job['Job Description'])
            
            # Vectorize job information
            job_skills_vector = self.skills_vectorizer.transform([job_skills])
            job_desc_vector = self.desc_vectorizer.transform([job_desc])
            
            # Calculate skill match score
            skill_match = cosine_similarity(user_skills_vector, job_skills_vector)[0][0]
            
            # Calculate education match score
            edu_match = self._calculate_education_match(
                user_profile['education'],
                job['Education']
            )
            
            # Calculate experience match score
            exp_match = self._calculate_experience_match(
                user_profile['experience'],
                job['Experience']
            )
            
            # Calculate category match score
            cat_match = 1.0 if 'preferred_industry' in user_profile and user_profile['preferred_industry'] == job['Category'] else 0.0
            
            # Calculate overall matching score with weights
            weights = {
                'skill': 0.45,  # Increased from 0.35
                'education': 0.20,  # Increased from 0.15
                'experience': 0.20,  # Increased from 0.15
                'category': 0.15  # Decreased from 0.20
            }
            
            matching_score = (
                weights['skill'] * skill_match +
                weights['education'] * edu_match +
                weights['experience'] * exp_match +
                weights['category'] * cat_match
            )
            
            recommendations.append({
                'job_title': job['Job Title'],
                'company': job['Company'],
                'job_description': job['Job Description'],
                'required_skills': job['skills'],
                'matching_score': matching_score,
                'skill_match_score': skill_match,
                'education_match_score': edu_match,
                'experience_match_score': exp_match,
                'category_match_score': cat_match
            })
        
        # Sort by matching score and return top N
        recommendations.sort(key=lambda x: x['matching_score'], reverse=True)
        return recommendations[:top_n]

def main():
    # Initialize the job recommender
    recommender = JobRecommender()
    
    # Example user profile
    user_profile = {
        'skills': "Python, Machine Learning, Data Analysis, SQL, Tableau",
        'education': "Master's Degree",
        'experience': "2-5 years",
        'preferred_industry': "Data Science"
    }
    
    # Print input user profile
    print("\nInput User Profile:")
    print("=" * 80)
    print(f"Skills: {user_profile['skills']}")
    print(f"Education: {user_profile['education']}")
    print(f"Experience: {user_profile['experience']}")
    print(f"Preferred Industry: {user_profile['preferred_industry']}")
    print("=" * 80)
    
    # Get recommendations
    recommendations = recommender.get_job_recommendations(user_profile)
    
    # Print recommendations
    print("\nTop Job Recommendations:")
    print("=" * 80)
    for i, rec in enumerate(recommendations, 1):
        print(f"\nRecommendation {i}")
        print("-" * 80)
        print(f"Title: {rec['job_title']}")
        print("\nDescription:")
        print(rec['job_description'])
        print("=" * 80)

if __name__ == "__main__":
    main()
