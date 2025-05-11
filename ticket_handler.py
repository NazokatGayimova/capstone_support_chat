import os
import requests

# Load GitHub credentials from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "NazokatGayimova/support-tickets")  # Update with your actual repo

def create_github_issue(name, email, question, answer):
    if not GITHUB_TOKEN:
        print("❌ GitHub token is missing. Set the GITHUB_TOKEN environment variable.")
        return False

    if not GITHUB_REPO:
        print("❌ GitHub repository name is missing. Set the GITHUB_REPO environment variable.")
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    # Construct the issue body
    issue_body = f"""
**User Name:** {name}  
**Email:** {email}  

**Question:**  
{question}  

**AI Response:**  
{answer}  
"""

    # Prepare the issue payload
    issue = {
        "title": f"[Support Request] {question[:40]}...",
        "body": issue_body.strip(),
        "labels": ["support"]
    }

    try:
        response = requests.post(url, json=issue, headers=headers)
        response.raise_for_status()
        print(f"✅ GitHub issue created successfully: {response.json().get('html_url')}")
        return True
    except requests.RequestException as e:
        print(f"❌ Failed to create GitHub issue: {e}")
        return False

