# Movie Recommendation System

## Project Description
Develop a movie recommendation system using collaborative filtering and content-based filtering techniques. The system will recommend movies to users based on their past interactions and movie preferences.

## Project Goals
1. **Data Collection:** Gather movie data from a reliable source (e.g., [MovieLens](https://grouplens.org/datasets/movielens/)).
2. **Data Preprocessing:** Clean and preprocess the data for analysis.
3. **Exploratory Data Analysis (EDA):** Perform EDA to understand the dataset and extract meaningful insights.
4. **Collaborative Filtering:** Implement collaborative filtering using matrix factorization techniques.
5. **Content-Based Filtering:** Implement content-based filtering using movie metadata (e.g., genres, directors, actors).
6. **Hybrid Recommendation System:** Combine collaborative and content-based filtering to build a hybrid recommendation system.
7. **Model Evaluation:** Evaluate the performance of the recommendation system using appropriate metrics.
8. **Deployment:** Deploy the recommendation system as a web application using Flask or Django.

## Tech Stack
- **Programming Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Surprise, Flask/Django
- **Tools:** Jupyter Notebook, Git, Docker (for containerization)
- **Database:** PostgreSQL/MySQL
- **Cloud Services:** AWS/GCP/Azure (optional for deployment)

## Project Structure
```
movie-recommendation-system/
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── exploratory_data_analysis.ipynb
│   ├── collaborative_filtering.ipynb
│   ├── content_based_filtering.ipynb
│   ├── hybrid_recommendation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── collaborative_filtering.py
│   ├── content_based_filtering.py
│   ├── hybrid_recommendation.py
│   ├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks for each step of the project.
4. Deploy the web application:
   ```bash
   python src/app.py
   ```

## Example Code Snippets

**Collaborative Filtering with SVD:**
```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate

# Load the dataset
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)

# Train the SVD algorithm
algo = SVD()
algo.fit(trainset)

# Test the algorithm
predictions = algo.test(testset)

# Evaluate the algorithm
accuracy.rmse(predictions)
```

**Content-Based Filtering:**
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the movie metadata
movies = pd.read_csv('data/movies.csv')

# Compute the TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Example usage
print(get_recommendations('Toy Story'))
```
