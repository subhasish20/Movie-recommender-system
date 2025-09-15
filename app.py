
from sentence_transformers import SentenceTransformer  # For sentence embeddings using a pre-trained transformer model
import pickle  
import pandas as pd  
import streamlit as st  
import requests 

# pre-trained(saved) model for sentence embeddings
model = SentenceTransformer('model/transformer_model')  # Path to the locally saved transformer model

# saved DataFrame and cosine similarity matrix using pickle
with open(r'model/bert_embeddings.pkl', 'rb') as f:  # Load embeddings of movie titles
    df = pickle.load(f)

with open(r'model/cosine_similarity.pkl', 'rb') as f:  # precomputed cosine similarity matrix
    cosine_sim = pickle.load(f)

# Function to fetch the movie poster from an external API (The Movie Database API)
def fetch_poster(movie_id):
    try:
        # Construct the URL to fetch the movie details (including the poster) from TMDB API
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()  # Fetch the movie details in JSON format
        poster_path = data.get('poster_path')  # Extract the poster path from the response
        
        # If no poster path is found, return a placeholder image
        if not poster_path:
            return "https://via.placeholder.com/500x750?text=No+Poster+Available"
        
        # If a poster is found, construct the full URL to fetch the poster image
        full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return full_path
    except Exception as e:
        # If there is an error (e.g., network issue), return a default error image
        return "https://via.placeholder.com/500x750?text=Error+Loading+Poster"

# Movie recommendation function based on cosine similarity
def recommend(movie, top_n=5):
    try:
        # Retrieve the index of the movie in the DataFrame based on the title
        index = df[df['title'] == movie].index[0]
    except IndexError:
        # If movie is not found, return an error message and empty lists
        st.error("Movie not found in the database.")
        return [], []  # Return empty lists if the movie is not found
    
    # Get the cosine similarity scores for the movie with all other movies
    similarity_scores = list(enumerate(cosine_sim[index]))
    
    # Sort the movies based on the similarity score in descending order (highest similarity first)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Prepare lists to store the recommended movie names and their poster URLs
    recommended_movie_names = []
    recommended_movie_posters = []
    
    # Add the top N recommended movies (starting from index 1 to exclude the movie itself)
    for i in similarity_scores[1:top_n+1]:
        movie_title = df.iloc[i[0]]['title']  # Get the movie title
        movie_id = df.iloc[i[0]]['movie_id']  # Get the movie ID for fetching the poster
        
        # Fetch the poster image URL using the movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movie_title)  # Add the movie title to the recommendation list
    
    # Fill the remaining slots with placeholder data if there are fewer than `top_n` recommendations
    while len(recommended_movie_names) < top_n:
        recommended_movie_names.append("No Recommendation Available")  # Add a placeholder name
        recommended_movie_posters.append("https://via.placeholder.com/500x750?text=No+Poster+Available")  # Add a placeholder poster
    
    return recommended_movie_names, recommended_movie_posters  # Return the recommendations and poster links

# Streamlit App User Interface
st.title('🎬 Movie Recommender System')  # Display the title of the app in a prominent manner

# Description and welcome message for the app
st.markdown(
    """
    ## Welcome to the Movie Recommender!
    Type or select a movie from the dropdown below and get personalized movie recommendations based on content similarity.
    Let's discover your next favorite movie! 🍿
    """
)

# Adding some extra space after the introduction for better UI layout
st.markdown("<br>", unsafe_allow_html=True)

# List of all available movie titles for the dropdown
movie_list = df['title'].values

# Dropdown menu to let the user select a movie from the list
selected_movie = st.selectbox(
    "Select a Movie",  # Prompt text for the dropdown
    movie_list,  # List of movie titles to choose from
    key="movie_select",  # Unique key to avoid conflicts with other Streamlit components
    help="Choose a movie from the list to get similar recommendations"  # Help text for the dropdown
)

# Adding some extra space after the dropdown for better UI layout
st.markdown("<br>", unsafe_allow_html=True)

# Custom button styling (red background, white text, etc.)
button_style = """
    <style>
    .stButton>button {
        background-color: #FF6347;  /* Tomato red color */
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 5px;
        padding: 12px 30px;
        width: 100%;  /* Full width button */
    }
    </style>
"""
# Apply custom button styling to the Streamlit app
st.markdown(button_style, unsafe_allow_html=True)

# When the user clicks the "Show Recommendation" button, get the recommended movies
if st.button('Show Recommendation'):
    # Get the list of recommended movie names and posters based on the selected movie
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

    # Display a subtitle for the recommended movies section
    st.subheader('Top Recommended Movies:')
    
    # Create columns dynamically based on the number of recommended movies (up to 5)
    cols = st.columns(min(5, len(recommended_movie_names)))  # Create 5 columns at maximum
    for idx, col in enumerate(cols):
        with col:
            # Display the movie poster in each column
            st.image(recommended_movie_posters[idx], use_column_width=True)
            # Display the movie title (in a smaller font size)
            st.text(recommended_movie_names[idx], font_size=14)
    
    # Add some extra space at the bottom for better layout
    st.markdown("<br><br>", unsafe_allow_html=True)
