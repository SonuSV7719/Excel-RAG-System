# Excel RAG System 📊

The **Excel RAG System** is a Retrieval-Augmented Generation (RAG) application that allows users to query Excel data using natural language. It leverages AI-powered Large Language Models (LLMs) to interpret user queries, generate SQL queries, and provide insightful answers based on the data.

## Features 🚀

- **Upload Excel Files**: Supports `.xlsx` and `.xls` formats.
- **Natural Language Queries**: Ask questions about your data in plain English.
- **AI-Powered SQL Generation**: Automatically generates SQL queries to retrieve relevant data.
- **Multiple LLM Providers**:
  - OpenAI (GPT-4o)
  - Google Gemini (2.0 Flash)
  - DeepSeek (via OpenRouter)
- **Interactive Data Preview**: View and explore your Excel sheets and their structure.
- **Question & Answer History**: Keeps track of your queries and responses.

## Installation 🛠️

1. Clone the repository:
   ```bash
   git clone https://github.com/SonuSV7719/Excel-RAG-System.git
   cd Excel-RAG-System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```

3. (Optional) Set up environment variables for API keys:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export GEMINI_API_KEY=your_google_gemini_api_key
   export OPENROUTER_API_KEY=your_openrouter_api_key
   ```

## Usage 📖

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the application in your browser (default: `http://localhost:8501`).

3. Upload an Excel file and select an LLM provider.

4. Enter your natural language query and get answers based on your data.

## Project Structure 📂

```
ExcelRAGSystem/
├── app.py                # Streamlit app for user interaction
├── database.py           # Handles Excel-to-SQLite database operations
├── llm_providers.py      # Implements LLM provider integrations
├── rag_system.py         # Core RAG system logic
├── requirement.txt       # Python dependencies
└── README.md             # Project documentation
```

## LLM Providers 🌐

### OpenAI (GPT-4o)
- Requires an API key from [OpenAI](https://platform.openai.com/).

### Google Gemini (2.0 Flash)
- Requires an API key from [Google MakerSuite](https://makersuite.google.com/).

### DeepSeek (via OpenRouter)
- Requires an API key from [OpenRouter](https://openrouter.ai/).

## Dependencies 📦

The project uses the following Python libraries:
- `streamlit`: For building the web interface.
- `pandas`: For data manipulation.
- `openpyxl`: For reading Excel files.
- `sqlite3`: For in-memory database operations.
- `openai`, `google-generativeai`, `requests`: For LLM integrations.

Refer to the [`requirement.txt`](requirement.txt) file for the full list of dependencies.

## Contributing 🤝

Contributions are welcome! If you'd like to improve this project, please fork the repository and submit a pull request.

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- [Streamlit](https://streamlit.io/) for the interactive web framework.
- [OpenAI](https://openai.com/), [Google Gemini](https://makersuite.google.com/), and [OpenRouter](https://openrouter.ai/) for their powerful LLM APIs.

---

**Excel RAG System** - Query your Excel data with the power of AI!
