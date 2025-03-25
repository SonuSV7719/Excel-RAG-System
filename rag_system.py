import pandas as pd

class RAGSystem:
    """
    Retrieval-Augmented Generation (RAG) system for querying Excel data using natural language.
    """
    
    def __init__(self, database, llm_provider):
        """
        Initialize the RAG system.
        
        Args:
            database (ExcelDatabase): Database instance
            llm_provider (LLMProvider): LLM provider instance
        """
        self.database = database
        self.llm_provider = llm_provider
    
    def query(self, natural_language_query):
        """
        Process a natural language query using the RAG system.
        
        Args:
            natural_language_query (str): Natural language query from the user
            
        Returns:
            tuple: (sql_query, response) where sql_query is the generated SQL and
                  response is the natural language answer
        """
        # Get database schema information to help the LLM understand the data
        database_schema = self._get_database_schema()
        
        # Generate SQL query from natural language
        sql_query = self.llm_provider.generate_sql(natural_language_query, database_schema)
        
        # Clean the SQL query (remove markdown code blocks if present)
        sql_query = self._clean_sql_query(sql_query)
        
        # Execute the SQL query
        try:
            results = self.database.execute_query(sql_query)
        except Exception as e:
            # If there's an error, try to fix the SQL query
            fixed_sql = self._fix_sql_query(sql_query, str(e), database_schema)
            # Clean the fixed SQL query as well
            fixed_sql = self._clean_sql_query(fixed_sql)
            results = self.database.execute_query(fixed_sql)
            sql_query = fixed_sql  # Update the SQL query to the fixed version
        
        # Generate a natural language response
        context = {
            'original_query': natural_language_query,
            'sql_query': sql_query,
            'schema': database_schema
        }
        response = self.llm_provider.generate_response(natural_language_query, results, context)
        
        return sql_query, response
    
    def _get_database_schema(self):
        """
        Get the database schema information.
        
        Returns:
            dict: Schema information about all tables in the database
        """
        sheet_info = self.database.get_all_sheet_info()
        schema = {}
        
        for sheet_name, info in sheet_info.items():
            table_name = info['clean_name']
            df = info['data']
            
            # Get column information
            columns = []
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                # Map pandas dtypes to more understandable types
                if 'int' in dtype_str:
                    column_type = 'INTEGER'
                elif 'float' in dtype_str:
                    column_type = 'REAL'
                elif 'datetime' in dtype_str:
                    column_type = 'DATETIME'
                else:
                    column_type = 'TEXT'
                
                columns.append({
                    'name': col,
                    'type': column_type,
                    'sample_values': df[col].dropna().sample(min(3, max(1, len(df)))).tolist()
                })
            
            schema[sheet_name] = {
                'table_name': table_name,
                'columns': columns,
                'row_count': len(df)
            }
        
        return schema
    
    def _fix_sql_query(self, sql_query, error_message, database_schema):
        """
        Attempt to fix an invalid SQL query using the LLM.
        
        Args:
            sql_query (str): Original SQL query
            error_message (str): Error message from the failed query
            database_schema (dict): Schema information about the database
            
        Returns:
            str: Fixed SQL query
        """
        schema_str = str(database_schema)
        
        # Create a prompt for the LLM to fix the SQL query
        prompt = f"""
        You are an expert in SQL. The following SQL query failed:
        
        ```sql
        {sql_query}
        ```
        
        Error message:
        {error_message}
        
        DATABASE SCHEMA:
        {schema_str}
        
        Please fix the SQL query to make it work with the given schema. Return ONLY the fixed SQL query without any explanations or comments.
        """
        
        # Generate the fixed SQL query
        fixed_sql = self.llm_provider.generate_sql(prompt, database_schema)
        
        return fixed_sql
        
    def _clean_sql_query(self, sql_query):
        """
        Clean a SQL query by removing markdown formatting.
        
        Args:
            sql_query (str): SQL query that may contain markdown formatting
            
        Returns:
            str: Cleaned SQL query
        """
        # Remove markdown code block syntax if present
        sql_query = sql_query.strip()
        
        # Remove ```sql at the beginning if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:].strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query[3:].strip()
            
        # Remove ``` at the end if present
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3].strip()
            
        return sql_query
