import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template
import difflib

# Initialize the Flask application
flask_app = Flask(__name__)

# Load the pre-trained model and similarity matrix
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))
df = pd.read_csv('movies.csv')

def get_recommendations(movie_name):
    list_of_all_titles = df['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if not find_close_match:
        return []

    close_match = find_close_match[0]
    index_of_the_movie = df[df.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended_movies = [df.iloc[movie[0]].title for movie in sorted_similar_movies[1:11]]
    return recommended_movies

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    movie_name = request.form.get('movie_name')
    recommendations = get_recommendations(movie_name)
    return render_template("recommendation.html", movie_name=movie_name, recommendations=recommendations)

if __name__ == "__main__":
    flask_app.run(debug=True)
