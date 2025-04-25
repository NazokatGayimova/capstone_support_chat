import requests
import os

# Store in env or directly here (not safe for public)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "your_github_token")
GITHUB_REPO = os.getenv("GITHUB_REPO", "username/repo")  # e.g., nazokat/support-tickets

def create_github_issue(name, email, question, answer):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    issue = {
        "title": f"[Support Request] {question[:40]}...",
        "body": f"**User Name:** {name}\n**Email:** {email}\n\n**Question:** {question}\n\n**AI Response:** {answer}",
        "labels": ["support"]
    }
    response = requests.post(url, json=issue, headers=headers)
    return response.status_code == 201
