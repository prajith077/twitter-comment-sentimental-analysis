import streamlit as st
import pickle
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load the vectorizer and model
try:
    with open(r'C:\Users\user\OneDrive\Desktop\ml project dataset\count_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    with open(r'C:\Users\user\OneDrive\Desktop\ml project dataset\logistic_regression.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

# Load dataset and specify the text column
dataset_path = r'C:\Users\user\OneDrive\Desktop\ml project dataset\COMMENTSS.csv'  # Replace with your dataset path
try:
    dataset = pd.read_csv(dataset_path)
    text_column = 'text'  # Replace with the actual name of your text column
    label_column = 'label'  # Replace with the actual name of your label column

    if text_column not in dataset.columns or label_column not in dataset.columns:
        st.error(f"The dataset does not contain the required columns: '{text_column}' or '{label_column}'.")
        st.stop()
    known_texts = dataset[text_column].tolist()
    known_labels = dataset[label_column].tolist()
    combined_texts = [f"{text} {label}" for text, label in zip(known_texts, known_labels)]
    known_texts_vectorized = vectorizer.transform(combined_texts)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Sentiment mapping
sentiment_map = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

# Function to resize the image to smaller dimensions
def resize_image(image_path, width, height):
    try:
        image = Image.open(image_path)
        aspect_ratio = image.width / image.height
        resized_image = image.resize((int(height * aspect_ratio), height))  # Maintain aspect ratio
        return resized_image
    except FileNotFoundError:
        st.warning("Image file not found.")
        return None

# Display the header image with caption
header_image_path = r'C:\Users\user\OneDrive\Desktop\ml project dataset\example.png'  # Replace with your image path
header_image = resize_image(header_image_path, width=60, height=60)  # Small dimensions
if header_image:
    st.image(header_image, caption="Twitter")

# App title and description
st.title("Tweet Analysis")

# Sidebar with additional info
st.sidebar.header("About")
st.sidebar.write(
    """
    This app predicts the emotional sentiment of your text. Possible sentiments include:
    - Sadness 
    - Joy 
    - Love 
    - Anger
    - Fear 
    - Surprise

    #### How to use:
    1. Enter a sentence in the text box.
    2. Click the Analyze Sentiment button.
    3. View the predicted sentiment below.
    """
)

# Input text
user_input = st.text_area("Type your sentence here:")

# Prediction button
if st.button("Analyze Sentiment"):
    if user_input.strip():  # Ensure input is not empty
        try:
            input_vectorized = vectorizer.transform([user_input])
            similarity_scores = cosine_similarity(input_vectorized, known_texts_vectorized)
            max_similarity_score = similarity_scores.max()

            # Set a similarity threshold (you can adjust this threshold as needed)
            similarity_threshold = 0.8

            if max_similarity_score >= similarity_threshold:
                prediction = model.predict(input_vectorized)[0]
                sentiment = sentiment_map.get(prediction, "Unknown")
                st.success(f"The sentiment is: {sentiment}")
            else:
                st.error("The input text does not match any known entries in the dataset.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Please enter a valid sentence.")
