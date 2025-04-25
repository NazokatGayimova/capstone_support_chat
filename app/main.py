import streamlit as st
from qa_engine import ask_question
from ticket_handler import create_github_issue

st.set_page_config(page_title="Volkswagen AI Chat", layout="centered")

st.markdown("""
    <style>
    .chat-user {
        background-color: #d7ebff;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    .chat-ai {
        background-color: #e5ffd7;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💬 Volkswagen AI Customer Support")

# Session state setup
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "ticket_offered" not in st.session_state:
    st.session_state.ticket_offered = False

# Define unclear answer logic
def is_unclear_response(answer: str, question: str):
    fallback_keywords = [
        "i don't know", "i do not have", "no specific information",
        "i'm not sure", "uncertain", "could not find", "not available"
    ]
    silly_question_keywords = [
        "rocket", "spaceship", "submarine", "airplane", "boat", "alien", "moon"
    ]

    answer_lower = answer.lower()
    question_lower = question.lower()

    return any(f in answer_lower for f in fallback_keywords) or any(k in question_lower for k in silly_question_keywords)

# Input form
with st.form(key="chat_form"):
    user_input = st.text_input("Ask a question about Volkswagen:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    with st.spinner("🤖 Thinking..."):
        answer = ask_question(user_input)
        st.session_state.last_question = user_input
        st.session_state.last_answer = answer
        st.session_state.ticket_offered = False

# Show the latest Q&A
if st.session_state.last_question:
    st.markdown(f"<div class='chat-user'>{st.session_state.last_question}</div>", unsafe_allow_html=True)

if st.session_state.last_answer:
    if not is_unclear_response(st.session_state.last_answer, st.session_state.last_question):
        st.markdown(f"<div class='chat-ai'>{st.session_state.last_answer}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-ai'>🤖 I'm not sure about that. You can submit a support ticket below for help from our team.</div>", unsafe_allow_html=True)

# Show support form only if answer is unclear
if st.session_state.last_answer:
    needs_ticket = is_unclear_response(st.session_state.last_answer, st.session_state.last_question)

    if needs_ticket and not st.session_state.ticket_offered:
        st.session_state.ticket_offered = True
        st.warning("You can submit your question as a support ticket below.")

        with st.form(key="ticket_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            issue_description = st.text_area("Any additional details?", placeholder="Optional message to support team...")
            send_ticket = st.form_submit_button("Create Support Ticket")

        if send_ticket:
            title = f"[Support] {st.session_state.last_question[:50]}..."
            body = f"""**User Name:** {name}
**Email:** {email}

**Question:** {st.session_state.last_question}

**AI Response:** {st.session_state.last_answer}

**User Comments:** {issue_description}
"""
            success = create_github_issue(name, email, title, body)
            if success:
                st.success("✅ Your support ticket was successfully created on GitHub!")
            else:
                st.error("❌ Failed to create the ticket. Please check your GitHub token or repo settings.")
