import streamlit as st
import pandas as pd
import os
from database import ExcelDatabase
from llm_providers import LLMProvider
from rag_system import RAGSystem

# Configure page
st.set_page_config(
    page_title="Excel RAG System",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("🔍 Excel RAG System")
st.markdown("""
Upload an Excel file and ask questions about your data in natural language.
The system will use AI to interpret your questions and provide answers based on your Excel data.
""")

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
if 'sheets' not in st.session_state:
    st.session_state.sheets = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# API key states
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'openrouter_api_key' not in st.session_state:
    st.session_state.openrouter_api_key = ""

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=['xlsx', 'xls'])

# Process the uploaded file
if uploaded_file and (st.session_state.uploaded_file != uploaded_file.name):
    try:
        st.session_state.uploaded_file = uploaded_file.name
        
        # Read all sheets from Excel file
        with st.spinner('Loading Excel file...'):
            st.session_state.sheets = pd.read_excel(uploaded_file, sheet_name=None)
            
            # Create the database
            st.session_state.db = ExcelDatabase()
            for sheet_name, df in st.session_state.sheets.items():
                st.session_state.db.add_sheet(sheet_name, df)
            
            st.success(f"Successfully loaded {len(st.session_state.sheets)} sheets from {uploaded_file.name}")
            
            # Reset chat history when a new file is uploaded
            st.session_state.chat_history = []
            
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        st.session_state.db = None
        st.session_state.sheets = None

# Display available sheets if file is uploaded
if st.session_state.sheets is not None:
    with st.expander("Available sheets and sample data"):
        sheet_names = list(st.session_state.sheets.keys())
        selected_sheet = st.selectbox("Select a sheet to preview:", sheet_names)
        
        if selected_sheet:
            st.dataframe(st.session_state.sheets[selected_sheet].head(5))
            
            # Show table schema
            st.subheader("Sheet Structure")
            df = st.session_state.sheets[selected_sheet]
            schema_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str)
            })
            st.table(schema_df)

    # LLM provider selection
    st.subheader("Select LLM Provider")
    llm_provider = st.selectbox(
        "Choose the AI model to use for answering your questions:",
        ["OpenAI (GPT-4o)", "Google (Gemini 2.0 Flash)", "DeepSeek (via OpenRouter)"]
    )
    
    # API key input based on selected provider
    api_key = None
    if "OpenAI" in llm_provider:
        api_key = st.text_input("Enter OpenAI API Key:", 
                               value=st.session_state.openai_api_key,
                               type="password",
                               help="Required for using OpenAI's GPT-4o model. Get it from platform.openai.com")
        st.session_state.openai_api_key = api_key
    elif "Google" in llm_provider:
        api_key = st.text_input("Enter Google Gemini API Key:", 
                               value=st.session_state.gemini_api_key,
                               type="password",
                               help="Required for using Google's Gemini 2.0 Flash model. Get it from makersuite.google.com")
        st.session_state.gemini_api_key = api_key
    else:  # OpenRouter
        api_key = st.text_input("Enter OpenRouter API Key:", 
                               value=st.session_state.openrouter_api_key,
                               type="password",
                               help="Required for using DeepSeek models via OpenRouter. Get it from openrouter.ai")
        st.session_state.openrouter_api_key = api_key

    # Query input
    st.subheader("Ask a question about your data")
    user_query = st.text_area("Enter your question:", height=100, 
                            placeholder="Example: What is the total sales in Quarter 1? Which product had the highest revenue?")

    # Process query
    if st.button("Submit Question"):
        if not user_query:
            st.warning("Please enter a question to proceed.")
        else:
            try:
                with st.spinner('Analyzing your question and searching through data...'):
                    # Check if API key is provided
                    if not api_key:
                        st.error(f"Please enter a valid API key for {llm_provider}")
                        st.stop()
                        
                    # Initialize the appropriate LLM provider with API key
                    if "OpenAI" in llm_provider:
                        provider = LLMProvider.create("openai", api_key)
                    elif "Google" in llm_provider:
                        provider = LLMProvider.create("gemini", api_key)
                    else:  # DeepSeek
                        provider = LLMProvider.create("openrouter", api_key)
                    
                    # Initialize the RAG system
                    rag = RAGSystem(st.session_state.db, provider)
                    
                    # Get response
                    sql_query, response = rag.query(user_query)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_query,
                        "sql": sql_query,
                        "answer": response,
                        "provider": llm_provider
                    })
                    
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Question & Answer History")
        total_questions = len(st.session_state.chat_history)
        for i, exchange in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"#### Question {total_questions - i}")
                st.markdown(f"**Q:** {exchange['question']}")
                st.markdown(f"**Using:** {exchange['provider']}")
                
                # with st.expander("View SQL Query"):
                #     st.code(exchange['sql'], language="sql")
                
                st.markdown(f"**Answer:** {exchange['answer']}")
                st.markdown("---")
else:
    st.info("Please upload an Excel file to begin.")

# Footer
st.markdown("---")
st.markdown("Excel RAG System - Query your Excel data using natural language")
