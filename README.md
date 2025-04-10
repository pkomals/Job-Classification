# Job Classification and Recommendation System
This project focuses on building a machine learning pipeline to classify job postings based on their descriptions and later extend the system for job recommendation based on user profiles.

## Phase 1: Job Classification

### Dataset
- The dataset used for this project is publicly available on Kaggle:  
[Job Description Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset/data).
- Imbalanced data with 147 unique job titles.

### Workflow

#### 1. Exploratory Data Analysis (EDA)
- Checked frequency distribution of job titles.
- Verified whether job titles appeared in job descriptions.
- Decided to work with top 10 most frequent job titles to reduce noise and overfitting.

#### 2. Data Preprocessing
- Removed duplicates and null values.
- Standard NLP steps:
  - Lowercasing
  - Removing punctuation and special characters
  - Stopword removal
  - Lemmatization

#### 3. Label Encoding
- Encoded job titles into numerical labels using `LabelEncoder`.

#### 4. Train-Test Split
- Stratified sampling to maintain class balance.
- 80-20 split on the filtered dataset.

#### 5. Text Vectorization
- Used `TfidfVectorizer` 

#### 6. Modeling
Implemented multiple ML models:
- **Naive Bayes**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **LSTM** (using custom embeddings, attention layer)

#### 7. Evaluation
Evaluated several machine learning models to classify job descriptions into the top 10 most frequent job titles using TF-IDF features. Below are the performance metrics:

| Model                               | Accuracy | Precision | Recall |
|-------------------------------------|----------|-----------|--------|
| Random Forest (RF)                  | 0.84     | 0.83      | 0.83   |
| Support Vector Machine (SVM)        | 0.84     | 0.83      | 0.83   |
| Naive Bayes (NB)                    | 0.79     | 0.76      | 0.75   |
| Feedforward Neural Network + BERT  | 0.8048   | —         | —      |
| LSTM (with TF-IDF)                  | 0.76     | 0.72      | 0.73   |

**LSTM** was not selected for the final model as it does not perform well with TF-IDF input due to the sequential nature of LSTM networks.
**SVM**, despite similar metrics to RF, showed signs of memorizing patterns in the training data, leading to increased variance and a risk of overfitting
**Random** Forest Classifier — chosen for its balance of high accuracy, interpretability, and reasonable training time.

---

## Phase 2: Job Recommendation

To recommend jobs to users based on their profile information(Content base filtering).

### Input
- **User Profile**: skills, education, work experience
- **Job Listings**: job title, description, required skills

### Output
- Recommended job titles

