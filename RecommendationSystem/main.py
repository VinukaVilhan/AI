from pydantic import validate_email
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# List of preferred activities (cleaned version from your input)
preferred_activities_list = [
    'cycling', 'historical monuments', 'village homestays', 'butterfly watching',
    'hot springs', 'wildlife viewing', 'sea cruises', 'themed parks', 'craft workshops',
    'fishing', 'sailing', 'history tours', 'literary tours', 'public art installations',
    'temple pilgrimages', 'architecture tours', 'golfing', 'hot air ballooning', 
    'spiritual retreats', 'cultural experiences', 'botanical gardens', 'boat safaris',
    'caving', 'mountain biking', 'camping', 'museum visits', 'turtle watching',
    'historic walks', 'safaris', 'waterfalls', 'scuba diving', 'elephant rides', 
    'bird watching', 'ayurvedic spa treatments', 'surfing', 'historic sites', 
    'art classes', 'traditional ceremonies', 'city tours', 'theater', 
    'amusement parks', 'architecture photography', 'beachfront dining', 'kayaking',
    'beach visits', 'rock climbing', 'arts and culture', 'snorkeling', 
    'animal encounters', 'archaeological sites', 'sailing lessons', 'whale watching',
    'local crafts', 'yoga retreats', 'cultural festivals', 'paddleboarding', 
    'horseback riding', 'ziplining', 'outdoor adventures', 'planetarium visits',
    'horse shows', 'water parks', 'photography', 'tea tasting', 'hiking', 
    'river cruises', 'sightseeing'
]

# Define the template for the Llama model focused on reviews
template = """
You are tasked with providing recommendations based on the following reviews of places in Sri Lanka: {place_reviews}.
Rules to follow,
1. Summarize the positive aspects mentioned in the reviews.
2. The recommendations should reflect the high points mentioned in the reviews.
3. Do not add any extra information. Just highlight positive attributes.
"""

# Initialize the Llama model
model = OllamaLLM(model="llama3.1")

# Create a chain
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def generate_review_summary_with_llama(place_reviews):
    response = chain.invoke({"place_reviews": place_reviews})
    # Handle response appropriately
    if hasattr(response, 'text'):
        return response.text
    else:
        return str(response)

# Load pre-trained BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, model = load_bert_model()

# Load datasets
@st.cache_data
def load_data():
    places_df = pd.read_csv('./input/updated_places_df.csv')
    visitors_df = pd.read_csv('./input/visitors_df_cleaned.csv')
    return places_df, visitors_df

places_df, visitors_df = load_data()

# Function to get BERT embeddings for a list of texts
def get_bert_embeddings(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)
        embeddings.append(sentence_embedding)
    return torch.stack(embeddings).squeeze(1)

# Combine relevant text data for places
places_df['combined_features'] = places_df['name'] + ' ' + places_df['formatted_address'] + ' ' + places_df['cleaned_reviews'] + ' ' + places_df['rating'].astype(str)

# Get BERT embeddings for places
@st.cache_data
def get_place_embeddings():
    return get_bert_embeddings(places_df['combined_features'].fillna(''))

place_embeddings = get_place_embeddings()

# Compute cosine similarity between user and place embeddings
def get_similarity_scores(user_embedding):
    return cosine_similarity(user_embedding.unsqueeze(0), place_embeddings).flatten()

def get_recommendations(user_preferences, top_n=5):
    # Generate recommendations using BERT model
    user_embedding = get_bert_embeddings([user_preferences])
    sim_scores = get_similarity_scores(user_embedding[0])
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    recommended_places = places_df.iloc[top_indices].copy()
    recommended_places.loc[:, 'similarity_score'] = sim_scores[top_indices]
    recommended_places.loc[:, 'combined_score'] = recommended_places['similarity_score'] * recommended_places['rating']
    recommended_places = recommended_places.sort_values('combined_score', ascending=False)

    # Generate reviews summary using Llama model
    reviews_text = " ".join(recommended_places['cleaned_reviews'].fillna(''))
    review_summary = generate_review_summary_with_llama(reviews_text)
    
    return recommended_places[['name', 'formatted_address', 'rating', 'similarity_score', 'combined_score']], review_summary

# Streamlit app
st.title('Personalized Place Recommendations for Sri Lanka')

# User input form
st.header('Enter your details')
user_name = st.text_input('Name')
user_email = st.text_input('Email Address')
preferred_activities = st.text_area('Preferred Activities')
bucket_list = st.text_area('Bucket List Destinations in Sri Lanka')

# Function to dynamically match place with user's preferred activities
def match_place_with_activity(place_name, place_address, user_activities):
    matched_activities = []
    
    # Check if any of the user's preferred activities match the place name or address
    for activity in preferred_activities_list:
        if activity.lower() in place_name.lower() or activity.lower() in place_address.lower():
            matched_activities.append(activity)
    
    # Check if user-specific activities (input from the form) match any general activities
    for activity in user_activities.split(','):
        activity = activity.strip().lower()
        if activity in preferred_activities_list:
            matched_activities.append(activity)
    
    # Create a reason based on matched activities
    if matched_activities:
        return f"This place is a great fit for your interests in {', '.join(set(matched_activities))}."
    else:
        return "This place aligns with your preferences, offering exciting activities in Sri Lanka."

# Your main recommendation logic
if st.button('Get Recommendations'):
    if user_name and validate_email(user_email) and preferred_activities and bucket_list:
        user_preferences = preferred_activities + ' ' + bucket_list
        recommendations, review_summary = get_recommendations(user_preferences)

        st.subheader(f"Personalized Recommendations for {user_name}")

        # Display detailed recommendations for each place
        for _, place in recommendations.iterrows():
            st.write(f"**Name:** {place['name']}")
            st.write(f"**Address:** {place['formatted_address']}")
            st.write(f"**Rating:** {place['rating']:.2f}")
            st.write(f"**Similarity Score:** {place['similarity_score']:.2f}")
            
            # Use the dynamic matching function
            recommendation_reason = match_place_with_activity(
                place['name'], 
                place['formatted_address'], 
                preferred_activities
            )
            
            # Display individual place recommendation
            st.write(f"**Why we recommend:** {recommendation_reason}")
            st.write("---")

        # Display overall review-based recommendations summary once
        st.subheader("Overall Recommendations Based on Reviews")
        st.write(review_summary)

        # Save user information
        new_visitor = pd.DataFrame({
            'User ID': [visitors_df['User ID'].max() + 1],
            'Name': [user_name],
            'Email Address': [user_email],
            'Preferred Activities': [preferred_activities],
            'Bucket List Destinations in Sri Lanka': [bucket_list],
            'Recommended Places': [', '.join(recommendations['name'].tolist())]
        })
        visitors_df = pd.concat([visitors_df, new_visitor], ignore_index=True)
        visitors_df.to_csv('path_to_visitors_df.csv', index=False)
        
        st.success("Recommendations generated and your information has been saved!")
    else:
        st.warning("Please fill in all fields correctly to get personalized recommendations.")
