# AI-Powered Communication Assistant Documentation
## Architecture
- **Backend**: Python with Flask for API, pandas for data processing, Hugging Face (DistilBERT for sentiment, T5 for response generation), SQLite for storage.
- **Frontend**: HTML/CSS/JavaScript with Chart.js for visualization.
- **Email Retrieval**: Simulated using the provided CSV (can be extended with Gmail API/IMAP).
- **Workflow**:
  1. Filter emails by subject keywords ("Support," "Query," "Request," "Help").
  2. Categorize emails by issue type using rule-based logic.
  3. Perform sentiment analysis with DistilBERT.
  4. Prioritize emails based on urgent keywords.
  5. Generate context-aware responses with T5.
  6. Store results in SQLite and display on a Flask dashboard.

## Approach
- **Filtering**: Used pandas to filter emails based on subject keywords.
- **Categorization**: Rule-based system for issue types (e.g., "Login Issue" for "unable to log into my account"). Could be enhanced with BERT for scalability.
- **Sentiment Analysis**: DistilBERT model classifies emails as Positive/Negative/Neutral.
- **Prioritization**: Urgent emails identified by keywords ("urgent," "critical") and sorted first.
- **Response Generation**: T5 generates responses based on issue type, sentiment, and email content. Prompts are engineered for professional, empathetic tone.
- **Dashboard**: Flask serves an HTML page with a table of emails, analytics, and a Chart.js bar chart for issue distribution.

## Setup Instructions
1. Install dependencies: `pip install pandas transformers flask torch sqlite3`.
2. Place `Sample_Support_Emails_Dataset.csv` in the project directory.
3. Create a `templates` folder with `dashboard.html`.
4. Run `python email_assistant.py` to start the server.
5. Access the dashboard at `http://localhost:5000`.

## Future Improvements
- Integrate Gmail API for real email retrieval.
- Use RAG with a knowledge base for more accurate responses.
- Add user authentication for secure dashboard access.

**Author**: Priyanshu Singh (priyanshusingh00004@gmail.com)