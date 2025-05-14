import streamlit as st
import requests
import json # For pretty printing JSON if needed

# Configuration
BACKEND_URL = "https://nl2sql-3p8r.onrender.com/api/nl-to-sql" # Make sure this matches your FastAPI backend URL



st.set_page_config(layout="wide", page_title="Natural Language to Redshift SQL")

st.title("️ Natural Language to Redshift SQL Converter ️")
st.markdown("""
Welcome! Enter your natural language query below, and we'll convert it to a Redshift SQL query using a powerful LLM.
If the (dummy) validation passes, the query will be stored in our MongoDB database.
""")

with st.form("nl_query_form"):
    natural_language_query = st.text_area(
        "Enter your Natural Language Query:", 
        height=100,
        placeholder="E.g., 'Show me all users who signed up last week and live in California'"
    )
    submit_button = st.form_submit_button(label="Convert to SQL")

if submit_button and natural_language_query.strip():
    with st.spinner("Converting your query to Redshift SQL..."):
        try:
            payload = {"natural_language_query": natural_language_query}
            response = requests.post(BACKEND_URL, json=payload, timeout=60) # Increased timeout for LLM
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            
            result = response.json()

            st.subheader("Conversion Result:")
            
            if result.get("error"):
                st.error(f"Backend Error: {result.get('message')}\nDetails: {result.get('error')}")
            else:
                st.success(result.get("message", "Processed successfully."))

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Natural Language Query:**")
                    st.text_area("", value=result.get("natural_language_query"), height=100, disabled=True, key="nl_display")
                
                with col2:
                    st.markdown("**Generated Redshift SQL Query:**")
                    st.code(result.get("generated_redshift_query", "No query generated."), language="sql")

                st.markdown(f"**LLM Model Used:** `{result.get('llm_model_used', 'N/A')}`")
                
                if result.get("stored_in_db"):
                    st.info("This query was successfully stored in the database.")
                else:
                    st.warning("This query was NOT stored in the database. Check backend logs if storage was expected.")
            
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend. Please ensure the backend server is running.")
        except requests.exceptions.Timeout:
            st.error("The request to the backend timed out. The LLM might be taking too long or the backend is overloaded.")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            try:
                error_detail = e.response.json()
                if "detail" in error_detail:
                    st.error(f"Detail: {error_detail['detail']}")
            except json.JSONDecodeError:
                pass # If the error response is not JSON, we've already shown the text
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

elif submit_button and not natural_language_query.strip():
    st.warning("Please enter a natural language query.")

st.markdown("---_---")
st.markdown("Schema for MongoDB entries:")
st.json({
    "natural_language_query": "string (the user's input)",
    "generated_redshift_query": "string (the SQL output from LLM)",
    "llm_model_used": "string (e.g., 'gemini-pro')",
    "is_valid_by_dummy_check": "boolean (True if dummy_validate_query passed)",
    "created_at": "datetime (UTC timestamp of entry creation)"
})

st.sidebar.header("About")
st.sidebar.info(
    "This application demonstrates a simple Natural Language to SQL system "
    "using Streamlit for the frontend, FastAPI for the backend, "
    "a Google Generative AI model (via Langchain) for NL-to-SQL conversion, "
    "and MongoDB for storing the generated queries."
)
st.sidebar.header("Instructions")
st.sidebar.markdown(
    "1. Ensure the FastAPI backend is running (`python backend/main.py`).\n"
    "2. Ensure MongoDB is running and accessible.\n"
    "3. Set your `GOOGLE_API_KEY` in `backend/.env`.\n"
    "4. Enter your query and click 'Convert to SQL'."
)
