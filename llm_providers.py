import os
import json
import requests
from abc import ABC, abstractmethod
from openai import OpenAI
import google.generativeai as genai

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    
    @staticmethod
    def create(provider_name, api_key=None):
        """
        Factory method to create an LLM provider.
        
        Args:
            provider_name (str): Name of the provider (openai, gemini, openrouter)
            api_key (str, optional): API key for the provider
            
        Returns:
            LLMProvider: An instance of the specified provider
        """
        if provider_name.lower() == "openai":
            return OpenAIProvider(api_key)
        elif provider_name.lower() == "gemini":
            return GeminiProvider(api_key)
        elif provider_name.lower() == "openrouter":
            return OpenRouterProvider(api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
    
    @abstractmethod
    def generate_sql(self, query, database_schema):
        """
        Generate SQL from a natural language query.
        
        Args:
            query (str): Natural language query
            database_schema (dict): Schema information about the database
            
        Returns:
            str: SQL query
        """
        pass
    
    @abstractmethod
    def generate_response(self, query, sql_result, context):
        """
        Generate a natural language response from SQL results.
        
        Args:
            query (str): Original natural language query
            sql_result (pandas.DataFrame): Result of the SQL query
            context (dict): Additional context information
            
        Returns:
            str: Natural language response
        """
        pass


class OpenAIProvider(LLMProvider):
    """
    OpenAI implementation of the LLM provider.
    """
    
    def __init__(self, api_key=None):
        """Initialize the OpenAI provider with API key."""
        # First check if API key is passed directly, then check environment
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        if not api_key:
            raise ValueError("OpenAI API key is required. Please provide it through the UI.")
            
        self.client = OpenAI(api_key=api_key)
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
    
    def generate_sql(self, query, database_schema):
        """Generate SQL from natural language using OpenAI."""
        schema_str = json.dumps(database_schema, indent=2)
        
        prompt = f"""
        You are an expert in translating natural language questions to SQL queries.
        
        DATABASE SCHEMA:
        {schema_str}
        
        USER QUESTION:
        {query}
        
        Your task is to generate a valid SQL query that answers the user's question based on the provided database schema.
        Return ONLY the SQL query without any explanations or comments. Make sure the SQL query is valid and uses the correct table and column names.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a SQL expert that translates natural language to SQL."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_response(self, query, sql_result, context):
        """Generate natural language response from SQL results using OpenAI."""
        # Convert the SQL result to a string representation
        if sql_result is not None and not sql_result.empty:
            result_str = sql_result.to_string()
        else:
            result_str = "No results found"
        
        prompt = f"""
        You are an expert data analyst providing answers based on SQL query results.
        
        USER QUESTION:
        {query}
        
        SQL QUERY RESULTS:
        {result_str}
        
        CONTEXT:
        {json.dumps(context, indent=2)}
        
        Your task is to provide a clear, concise answer to the user's question based on the SQL query results.
        Explain the results in natural language, providing insights and context where appropriate.
        If the results are empty or don't directly answer the question, acknowledge this and explain possible reasons why.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data analysis expert explaining query results in natural language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()


class GeminiProvider(LLMProvider):
    """
    Google Gemini implementation of the LLM provider.
    """
    
    def __init__(self, api_key=None):
        """Initialize the Gemini provider with API key."""
        # First check if API key is passed directly, then check environment
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            raise ValueError("Gemini API key is required. Please provide it through the UI.")
            
        # Store API key for later use with each request
        self.api_key = api_key
    
    def generate_sql(self, query, database_schema):
        """Generate SQL from natural language using Gemini."""
        schema_str = json.dumps(database_schema, indent=2)
        
        prompt = f"""
        You are an expert in translating natural language questions to SQL queries.
        
        DATABASE SCHEMA:
        {schema_str}
        
        USER QUESTION:
        {query}
        
        Your task is to generate a valid SQL query that answers the user's question based on the provided database schema.
        Return ONLY the SQL query without any explanations or comments. Make sure the SQL query is valid and uses the correct table and column names.
        """
        
        # Configure the generative model
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate the SQL
        response = model.generate_content(prompt)
        
        return response.text.strip()
    
    def generate_response(self, query, sql_result, context):
        """Generate natural language response from SQL results using Gemini."""
        # Convert the SQL result to a string representation
        if sql_result is not None and not sql_result.empty:
            result_str = sql_result.to_string()
        else:
            result_str = "No results found"
        
        prompt = f"""
        You are an expert data analyst providing answers based on SQL query results.
        
        USER QUESTION:
        {query}
        
        SQL QUERY RESULTS:
        {result_str}
        
        CONTEXT:
        {json.dumps(context, indent=2)}
        
        Your task is to provide a clear, concise answer to the user's question based on the SQL query results.
        Explain the results in natural language, providing insights and context where appropriate.
        If the results are empty or don't directly answer the question, acknowledge this and explain possible reasons why.
        """
        
        # Configure the generative model
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate the response
        response = model.generate_content(prompt)
        
        return response.text.strip()


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter implementation for DeepSeek model.
    """
    
    def __init__(self, api_key=None):
        """Initialize the OpenRouter provider with API key."""
        # First check if API key is passed directly, then check environment
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            
        if not api_key:
            raise ValueError("OpenRouter API key is required. Please provide it through the UI.")
            
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "deepseek/deepseek-r1-zero:free"  # Updated to latest model
    
    def generate_sql(self, query, database_schema):
        """Generate SQL from natural language using DeepSeek via OpenRouter."""
        schema_str = json.dumps(database_schema, indent=2)
        
        prompt = f"""
        You are an expert in translating natural language questions to SQL queries.
        
        DATABASE SCHEMA:
        {schema_str}
        
        USER QUESTION:
        {query}
        
        Your task is to generate a valid SQL query that answers the user's question based on the provided database schema.
        Return ONLY the SQL query without any explanations or comments. Make sure the SQL query is valid and uses the correct table and column names.
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a SQL expert that translates natural language to SQL."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"].strip()
    
    def generate_response(self, query, sql_result, context):
        """Generate natural language response from SQL results using DeepSeek via OpenRouter."""
        # Convert the SQL result to a string representation
        if sql_result is not None and not sql_result.empty:
            result_str = sql_result.to_string()
        else:
            result_str = "No results found"
        
        prompt = f"""
        You are an expert data analyst providing answers based on SQL query results.
        
        USER QUESTION:
        {query}
        
        SQL QUERY RESULTS:
        {result_str}
        
        CONTEXT:
        {json.dumps(context, indent=2)}
        
        Your task is to provide a clear, concise answer to the user's question based on the SQL query results.
        Explain the results in natural language, providing insights and context where appropriate.
        If the results are empty or don't directly answer the question, acknowledge this and explain possible reasons why.
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a data analysis expert explaining query results in natural language."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"].strip()
