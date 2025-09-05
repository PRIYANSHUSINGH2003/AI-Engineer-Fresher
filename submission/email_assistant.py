import pandas as pd
from transformers import pipeline
from flask import Flask, render_template
import sqlite3
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize Hugging Face models (load once globally)
try:
    logger.info("Loading Hugging Face models...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    response_generator = pipeline("text2text-generation", model="t5-small")
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

# Load dataset (simulating email retrieval)
try:
    df = pd.read_csv("68b1acd44f393_Sample_Support_Emails_Dataset.csv")
    df["sent_date"] = pd.to_datetime(df["sent_date"])
    logger.info("Dataset loaded successfully.")
except FileNotFoundError:
    logger.error("CSV file not found. Ensure '68b1acd44f393_Sample_Support_Emails_Dataset.csv' is in the project directory.")
    raise

# Filter emails by subject keywords
def filter_emails(df):
    keywords = ["support", "query", "request", "help"]
    return df[df["subject"].str.lower().str.contains("|".join(keywords), na=False)]

# Categorize emails by issue type
def categorize_issue(body):
    body = body.lower()
    if "unable to log into my account" in body:
        return "Login Issue"
    elif "third-party APIs" in body or "CRM integration" in body:
        return "API/CRM Integration"
    elif "servers are down" in body:
        return "Server Downtime"
    elif "billing error" in body or "charged twice" in body:
        return "Billing Error"
    elif "reset my password" in body:
        return "Password Reset Failure"
    elif "verification email never arrived" in body:
        return "Account Verification"
    elif "requesting a refund" in body:
        return "Refund Process"
    elif "system is completely inaccessible" in body:
        return "System Inaccessibility"
    elif "pricing tiers" in body:
        return "Pricing Inquiry"
    return "Other"

# Prioritize emails based on subject keywords
def prioritize_email(subject):
    urgent_keywords = ["urgent", "critical", "immediate"]
    return "Urgent" if any(kw in subject.lower() for kw in urgent_keywords) else "Not Urgent"

# Generate context-aware responses (optimized)
def generate_response(row):
    try:
        issue = row["issue_type"]
        sentiment = row["sentiment"]
        body = row["body"][:200]  # Limit input length for faster processing
        prompt = f"Generate a professional, empathetic email response for a {issue} issue. Sentiment: {sentiment}. Email content: {body}"
        response = response_generator(prompt, max_new_tokens=50)[0]["generated_text"]
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Error generating response."

# Process emails
def process_emails():
    try:
        filtered_df = filter_emails(df).copy()
        filtered_df.loc[:, "issue_type"] = filtered_df["body"].apply(categorize_issue)
        filtered_df.loc[:, "sentiment"] = filtered_df["body"].apply(lambda x: sentiment_analyzer(x)[0]["label"])
        filtered_df.loc[:, "priority"] = filtered_df["subject"].apply(prioritize_email)
        filtered_df.loc[:, "response"] = filtered_df.apply(generate_response, axis=1)
        
        # Store in SQLite
        conn = sqlite3.connect("emails.db")
        filtered_df.to_sql("emails", conn, if_exists="replace", index=False)
        conn.close()
        logger.info("Emails processed and saved to SQLite.")
        return filtered_df
    except Exception as e:
        logger.error(f"Error processing emails: {e}")
        raise

# Analytics for dashboard
def get_analytics(df):
    try:
        reference_date = datetime(2025, 8, 26, 23, 59, 59)
        last_24h = reference_date - timedelta(hours=24)
        recent_emails = df[df["sent_date"] >= last_24h]
        return {
            "total_emails": len(recent_emails),
            "resolved_emails": 0,
            "pending_emails": len(recent_emails),
            "issue_counts": df["issue_type"].value_counts().to_dict(),
            "sentiment_counts": df["sentiment"].value_counts().to_dict(),
            "priority_counts": df["priority"].value_counts().to_dict()
        }
    except Exception as e:
        logger.error(f"Error generating analytics: {e}")
        return {}

# Flask route for dashboard
@app.route("/")
def dashboard():
    try:
        df = process_emails()
        analytics = get_analytics(df)
        return render_template("dashboard.html", emails=df.to_dict("records"), analytics=analytics)
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return f"Error: {e}", 500

# Main execution
if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(host="0.0.0.0", port=5001, debug=True)