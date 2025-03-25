import pandas as pd
import sqlite3

class ExcelDatabase:
    """
    A class to handle Excel database operations.
    Converts Excel sheets to SQLite tables in memory.
    """
    
    def __init__(self):
        """Initialize the in-memory SQLite database."""
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.sheets = {}
        
    def add_sheet(self, sheet_name, df):
        """
        Add a DataFrame as a table in the SQLite database.
        
        Args:
            sheet_name (str): Name of the sheet/table
            df (pandas.DataFrame): DataFrame to add
        """
        # Clean the table name to ensure it's valid for SQLite
        clean_name = self._clean_table_name(sheet_name)
        
        # Store the mapping between original and clean names
        self.sheets[sheet_name] = {
            'clean_name': clean_name,
            'columns': list(df.columns),
            'data': df
        }
        
        # Write the DataFrame to the SQLite database
        df.to_sql(clean_name, self.conn, index=False, if_exists='replace')
    
    def execute_query(self, query):
        """
        Execute an SQL query on the SQLite database.
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            pandas.DataFrame: Result of the query
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_all_sheet_info(self):
        """
        Get information about all sheets in the database.
        
        Returns:
            dict: Information about all sheets
        """
        return self.sheets
    
    def get_table_schema(self, table_name):
        """
        Get the schema for a specific table.
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            list: Column information for the table
        """
        clean_name = self._clean_table_name(table_name)
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({clean_name})")
        return cursor.fetchall()
    
    def _clean_table_name(self, name):
        """
        Clean a table name to ensure it's valid for SQLite.
        
        Args:
            name (str): Original table name
            
        Returns:
            str: Cleaned table name
        """
        # Replace spaces and special characters with underscores
        import re
        clean = re.sub(r'[^a-zA-Z0-9]', '_', name)
        
        # Ensure it doesn't start with a number
        if clean[0].isdigit():
            clean = 'sheet_' + clean
            
        return clean
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
