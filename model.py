import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('movies.csv')

# Fill missing values
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    df[feature] = df[feature].fillna('')

# Combine features into a single string
combined_features = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']

# Vectorize the combined features
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute the cosine similarity
similarity = cosine_similarity(feature_vectors)

# Save the vectorizer and similarity matrix
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))
