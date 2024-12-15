import streamlit as st
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("API Key not found. Please check your .env file.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=api_key)

# # Load the dataset
# df = pd.read_csv('../Data/Dataset.csv',encoding='latin1')
############################################################

# Set the Streamlit app directory
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data'))

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", app_dir)

# Load the dataset using the correct relative path
df = pd.read_csv(os.path.join(app_dir, 'Dataset.csv'), encoding='latin1')
##############################################################

def create_prompt(question):
    return f'''
    Using the dataset:\n{df}\nAnswer the question: {question}.
    '''

def query_groq(prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an insightful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            # model="llama-3.2-11b-vision-preview",
            # model="llama3-8b-8192",
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def queries_page():
    # st.title("Queries")

    # Create a container for the messages
    messages = st.container()

    # Capture user input using st.chat_input
    question = st.chat_input("Ask a question:")

    # If the user enters a question and presses enter
    if question:
        # Display user message
        with messages:
            st.chat_message("user").write(question)

        # Generate the assistant's response based on the user's question
        response = query_groq(create_prompt(question))  # Your query logic

        # Display assistant's response
        with messages:
            st.chat_message("assistant").write(response)

    # else:
    #     # If no input is entered, display an error
    #     st.error("Please enter a valid question.")

