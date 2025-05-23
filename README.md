# Natural Language to Redshift SQL Application

This is a full-stack application that converts natural language queries into AWS Redshift SQL queries. It uses a Large Language Model (LLM) via Langchain and Google Generative AI for the conversion, FastAPI for the backend, Streamlit for the frontend, and MongoDB to store the generated queries.

## Project Structure

```
nl2sql_app/
├── backend/
│   ├── main.py                 # FastAPI application logic
│   ├── requirements.txt        # Backend Python dependencies
│   └── .env.example            # Example environment variables for backend
├── frontend/
│   ├── app.py                  # Streamlit application logic
│   └── requirements.txt        # Frontend Python dependencies
├── .gitignore                  # Specifies intentionally untracked files
└── README.md                   # This file
```

## Features

-   **Natural Language to SQL:** Converts plain English queries into Redshift SQL.
-   **LLM Integration:** Utilizes `langchain-google-genai` (specifically the `gemini-pro` model by default) for query generation.
-   **Backend API:** A FastAPI backend handles the core logic, including LLM interaction and database operations.
-   **Interactive Frontend:** A Streamlit frontend provides a user-friendly interface to input queries and view results.
-   **MongoDB Storage:** Stores the natural language query, the generated Redshift SQL, the LLM model used, and a timestamp in a MongoDB collection after a dummy validation step.

## Prerequisites

-   Python 3.9+
-   MongoDB instance running and accessible.
-   A Google API Key with access to the Generative Language API (e.g., for Gemini models).

## Setup and Running the Application

Follow these steps to set up and run the application:

### 1. Clone the Repository (if applicable)

If you have this project in a Git repository, clone it first:
```bash
git clone https://github.com/ansuman-shukla/nl2sql.git
cd nl2sql
```

### 2. Backend Setup

The backend is a FastAPI application.

   a. **Navigate to the backend directory:**
      ```powershell
      cd backend
      ```

   b. **Create and activate a virtual environment:**
      ```powershell
      python -m venv venv_backend
      .\venv_backend\Scripts\activate 
      ```
      *For other operating systems (Linux/macOS):*
      ```bash
      python3 -m venv venv_backend
      source venv_backend/bin/activate
      ```

   c. **Install backend dependencies:**
      ```powershell
      pip install -r requirements.txt
      ```

   d. **Set up environment variables:**
      - Rename `.env.example` to `.env` (if it exists in the `backend` folder, otherwise create it).
      - Open the `.env` file and add your `GOOGLE_API_KEY`.
      - Update `MONGO_URI`, `MONGO_DB_NAME`, and `MONGO_COLLECTION_NAME` if they differ from the defaults.
      ```env
      GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
      MONGO_URI="mongodb://localhost:27017/"
      MONGO_DB_NAME="nl2sql_db"
      MONGO_COLLECTION_NAME="queries"
      # LLM_MODEL_NAME="gemini-pro" # Optional: uncomment to use a different model
      ```

   e. **Run the backend server:**
      ```powershell
      python main.py
      ```
      The backend server will typically start on `http://localhost:8000`.

### 3. Frontend Setup

The frontend is a Streamlit application.

   a. **Navigate to the frontend directory (from the project root):**
      ```powershell
      cd ..\frontend 
      ```
      *(If you are still in the `backend` directory)*

      Or from the project root:

      ```powershell
      cd frontend
      ```

   b. **Create and activate a virtual environment:**
      ```powershell
      python -m venv venv_frontend
      .\venv_frontend\Scripts\activate
      ```
      *For other operating systems (Linux/macOS):*
      ```bash
      python3 -m venv venv_frontend
      source venv_frontend/bin/activate
      ```

   c. **Install frontend dependencies:**
      ```powershell
      pip install -r requirements.txt
      ```

   d. **Run the frontend application:**
      ```powershell
      streamlit run app.py
      ```
      The Streamlit application will typically open in your browser at `http://localhost:8501`.

## API Endpoint

The backend exposes the following API endpoint:

-   **`POST /api/nl-to-sql`**
    -   **Request Body:**
        ```json
        {
            "natural_language_query": "Your natural language query here"
        }
        ```
    -   **Response Body (Success):**
        ```json
        {
            "natural_language_query": "string",
            "generated_redshift_query": "string",
            "stored_in_db": true,
            "message": "string",
            "llm_model_used": "string"
        }
        ```
    -   **Response Body (Error):**
        ```json
        {
            "natural_language_query": "string",
            "generated_redshift_query": null,
            "stored_in_db": false,
            "message": "string",
            "llm_model_used": null,
            "error": "string"
        }
        ```

## MongoDB Schema

Entries stored in MongoDB will follow this schema:

```json
{
    "_id": "ObjectId(...)", // Automatically generated by MongoDB
    "natural_language_query": "The original natural language query from the user.",
    "generated_redshift_query": "The Redshift SQL query generated by the LLM.",
    "llm_model_used": "The name of the LLM model used (e.g., 'gemini-pro').",
    "is_valid_by_dummy_check": true, // Result of the dummy validation function
    "created_at": "ISODate(...)" // UTC timestamp of when the entry was created
}
```

## How it Works

1.  The user enters a natural language query in the Streamlit frontend.
2.  Streamlit sends this query to the FastAPI backend.
3.  The FastAPI backend uses the Langchain library with the `ChatGoogleGenerativeAI` LLM to convert the natural language query into a Redshift SQL query. A specific prompt is used to guide the LLM.
4.  The generated SQL query undergoes a `dummy_validate_query` function (which, as per requirements, always returns `True`).
5.  If the dummy validation passes, the original query, the generated SQL, the LLM model name, and a timestamp are stored as a document in the specified MongoDB collection.
6.  The backend returns the result (including the generated query and storage status) to the Streamlit frontend, which then displays it to the user.
