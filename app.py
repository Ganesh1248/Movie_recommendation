import numpy as np
import pandas as pd
import ast
import nltk
import requests
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
def load_data():
    movies = pd.read_csv('data/tmdb_5000_movies.csv.zip')
    credits = pd.read_csv('data/tmdb_5000_credits.csv.zip')
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    return movies

# Preprocess Data
def preprocess_data(movies):
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L

    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])
    
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    return new_df

# Train Model
def train_model(new_df):
    ps = PorterStemmer()
    def stem(text):
        return " ".join([ps.stem(i) for i in text.split()])
    
    new_df['tags'] = new_df['tags'].apply(stem)
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return new_df, similarity

# Fetch Movie Poster
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url).json()
    poster_path = data.get('poster_path', '')
    return "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else ""

# Recommendation Function
def recommend(movie, new_df, similarity, movies):
    if movie not in new_df['title'].values:
        return ["Movie not found"], []
    
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))
    
    return recommended_movie_names, recommended_movie_posters

# Streamlit UI
def main():
    st.title("Movie Recommendation System")
    
    movies = load_data()
    new_df = preprocess_data(movies)
    new_df, similarity = train_model(new_df)
    
    movie_list = new_df['title'].tolist()
    selected_movie = st.selectbox("Select a movie:", movie_list)
    
    if st.button("Recommend"):
        recommendations, posters = recommend(selected_movie, new_df, similarity, movies)
        
        cols = st.columns(5)
        for col, (rec, poster) in zip(cols, zip(recommendations, posters)):
            with col:
                st.text(rec)
                if poster:
                    st.image(poster)

if __name__ == "__main__":
    main()
