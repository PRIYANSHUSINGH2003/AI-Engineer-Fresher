import pandas as pd

# Load dataset
df = pd.read_csv("68b1acd44f393_Sample_Support_Emails_Dataset.csv")

# Define issue categories based on keywords in body
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

# Apply categorization
df["issue_type"] = df["body"].apply(categorize_issue)

# Save categorized dataset
df.to_csv("categorized_emails.csv", index=False)

# Print summary
print("Issue Type Distribution:")
print(df["issue_type"].value_counts())

# Optional: Prioritize emails with urgent keywords
df["priority"] = df["subject"].str.lower().apply(
    lambda x: "High" if any(kw in x for kw in ["urgent", "critical", "immediate"]) else "Normal"
)
print("\nPriority Distribution:")
print(df["priority"].value_counts())