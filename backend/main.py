import os
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "nl2sql_db")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "queries")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in the .env file.")

# Initialize FastAPI app
app = FastAPI(title="Natural Language to Redshift SQL API")

# Initialize MongoDB client
try:
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    # Test connection
    client.admin.command('ping') 
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # Depending on the desired behavior, you might want to exit or handle this differently
    # For now, we'll let the app start but log the error. Operations requiring DB will fail.
    db = None
    collection = None


# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME, 
        google_api_key=GOOGLE_API_KEY, 
        temperature=0.2,
    )
    
    prompt_template_text = """You are an expert in AWS Redshift SQL.
Your task is to convert the following natural language query into a syntactically correct AWS Redshift SQL query.
Only output the SQL query itself. Do not include any explanations, markdown formatting, or any text other than the SQL query.

Natural Language Query:
{natural_language_query}

Redshift SQL Query:"""
    
    prompt = PromptTemplate(
        input_variables=["natural_language_query"],
        template=prompt_template_text
    )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    print(f"LLM ({LLM_MODEL_NAME}) initialized successfully.")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm_chain = None # App can start, but LLM functionality will be impaired.

# Pydantic Models
class NLQueryRequest(BaseModel):
    natural_language_query: str

class SQLQueryResponse(BaseModel):
    natural_language_query: str
    generated_redshift_query: str | None = None
    stored_in_db: bool
    message: str
    llm_model_used: str | None = None
    error: str | None = None

# Dummy validation function
def dummy_validate_query(query: str) -> bool:
    """
    A dummy function that always returns True.
    In a real scenario, this could contain actual validation logic.
    """
    print(f"Dummy validation called for query: {query[:100]}...") # Log first 100 chars
    return True

@app.post("/api/nl-to-sql", response_model=SQLQueryResponse)
async def nl_to_sql(request: NLQueryRequest = Body(...)):
    """
    Converts a natural language query to a Redshift SQL query using an LLM,
    validates it (dummy validation), and stores it in MongoDB.
    """
    if not llm_chain:
        raise HTTPException(status_code=500, detail="LLM not initialized. Check server logs.")

    nl_query = request.natural_language_query
    if not nl_query.strip():
        raise HTTPException(status_code=400, detail="Natural language query cannot be empty.")

    try:
        # Generate Redshift query using LLM
        response = llm_chain.invoke({"natural_language_query": nl_query})
        redshift_query = response['text'].strip()
        
        # Clean up potential markdown backticks if the LLM adds them
        if redshift_query.startswith("```sql"):
            redshift_query = redshift_query[len("```sql"):].strip()
        if redshift_query.startswith("```"):
            redshift_query = redshift_query[len("```"):].strip()
        if redshift_query.endswith("```"):
            redshift_query = redshift_query[:-len("```")].strip()

    except Exception as e:
        print(f"Error during LLM query generation: {e}")
        return SQLQueryResponse(
            natural_language_query=nl_query,
            stored_in_db=False,
            message="Error generating Redshift query.",
            error=str(e)
        )

    # Dummy validation
    is_valid = dummy_validate_query(redshift_query)
    stored_in_db = False
    
    if is_valid:
        if collection is not None:
            try:
                entry = {
                    "natural_language_query": nl_query,
                    "generated_redshift_query": redshift_query,
                    "llm_model_used": LLM_MODEL_NAME,
                    "is_valid_by_dummy_check": True,
                    "created_at": datetime.utcnow()
                }
                collection.insert_one(entry)
                stored_in_db = True
                message = "Redshift query generated and stored successfully."
            except Exception as e:
                print(f"Error storing data in MongoDB: {e}")
                message = "Redshift query generated but failed to store in MongoDB."
                return SQLQueryResponse(
                    natural_language_query=nl_query,
                    generated_redshift_query=redshift_query,
                    stored_in_db=False,
                    message=message,
                    llm_model_used=LLM_MODEL_NAME,
                    error=str(e)
                )
        else:
            message = "Redshift query generated but MongoDB is not available for storage."
            print("MongoDB collection is not available. Skipping storage.")
    else:
        # This part is currently unreachable as dummy_validate_query always returns True
        message = "Redshift query generated but failed dummy validation (this should not happen)."

    return SQLQueryResponse(
        natural_language_query=nl_query,
        generated_redshift_query=redshift_query,
        stored_in_db=stored_in_db,
        message=message,
        llm_model_used=LLM_MODEL_NAME
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting backend server...")
    print(f"Ensure GOOGLE_API_KEY is set and MongoDB is running at {MONGO_URI}.")
    print(f"Using LLM: {LLM_MODEL_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
