import streamlit as st
import pandas as pd
import sqlite3
from core.generation import askAI

data = None

# Stub for the bot response function
def getResponse(user_input, model_name, db_path):
    # Use the askAI function to get the bot's response
    return askAI(model_name, db_path, user_input)

# Function to load a CSV file
def load_csv(file):
    return pd.read_csv(file)

# Function to load a SQLite or DB file and display its tables
def load_db(file):
    conn = sqlite3.connect(file)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(query, conn)
    return tables


# Streamlit UI
def app():
    st.title("File Upload and Chat with Bot")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV, SQLite, or DB file", type=["csv", "sqlite", "db"])

    # Check the file type and load data accordingly
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            data = load_csv(uploaded_file)
            st.write("CSV Data", data)
        elif uploaded_file.type in ["application/sqlite", "application/x-sqlite3", "application/db", "application/octet-stream"]:
            with open("temp.db", "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
            db_path = "temp.db"
            tables = load_db(db_path)
            st.write("SQLite Tables", tables)
        else:
            st.error("Unsupported file type. Please upload a CSV or SQLite/DB file.")

    # Model selection
        model_options = ["NyanDoggo/Qwen2.5-Coder-0.5B-Instruct-Spider-Reasoning", "qwen2.5-coder:0.5b-base-q2_K"]
        selected_model = st.selectbox("Select AI Model", model_options)

    # Chat interface with bot
    user_input = st.text_input("Ask the bot:")
    if user_input:
        bot_response = getResponse(user_input, selected_model, db_path)
        st.write("Bot Response:", bot_response)

    # Table output (example placeholder table)
    # st.write("Sample Output Table:")
    # example_data = pd.DataFrame({
    #     'Column 1': [1, 2, 3],
    #     'Column 2': ['A', 'B', 'C'],
    #     'Column 3': [10.5, 20.3, 30.7]
    # })
    # st.table(example_data)


if __name__ == "__main__":
    app()
