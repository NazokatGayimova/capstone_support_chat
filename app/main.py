# app/main.py

import streamlit as st
from app.qa_engine import ask_question, submit_support_ticket

# Streamlit UI configuration
st.set_page_config(page_title="Volkswagen AI Customer Support", page_icon="🚗")

if "history" not in st.session_state:
    st.session_state.history = []

if "ticket_mode" not in st.session_state:
    st.session_state.ticket_mode = False

st.title("🚗 Volkswagen AI Customer Support")
st.write("Ask a question about Volkswagen:")

user_input = st.text_input("Type your question here:")

if user_input:
    try:
        answer = ask_question(user_input)
        st.session_state.history.append(("User", user_input))
        st.session_state.history.append(("Assistant", answer))
        
        st.success(answer)
        
        if "not sure" in answer.lower() or "cannot find" in answer.lower():
            st.session_state.ticket_mode = True

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

# Conversation history
if st.session_state.history:
    st.subheader("🧵 Conversation History")
    for role, message in st.session_state.history:
        if role == "User":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Assistant:** {message}")

# If unclear answer, suggest ticket submission
if st.session_state.ticket_mode:
    st.subheader("📩 Submit a Support Ticket")
    with st.form("support_ticket_form"):
        user_name = st.text_input("Your Name")
        user_email = st.text_input("Your Email")
        issue_summary = st.text_input("Issue Summary (Title)")
        issue_description = st.text_area("Describe the Issue")
        submitted = st.form_submit_button("Submit Ticket")
        
        if submitted:
            ticket_response = submit_support_ticket(user_name, user_email, issue_summary, issue_description)
            st.success(ticket_response)
            st.session_state.ticket_mode = False

