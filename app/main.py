import streamlit as st
from qa_engine import ask_question, submit_support_ticket

st.set_page_config(page_title="Volkswagen AI Customer Support", page_icon="🚗")
st.title("💬 Volkswagen AI Customer Support")
st.write("Ask a question about Volkswagen:")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "awaiting_ticket" not in st.session_state:
    st.session_state.awaiting_ticket = False

user_input = st.text_input("Type your question here:")

if user_input:
    st.session_state.conversation.append(("User", user_input))
    st.session_state.last_question = user_input

    answer = ask_question(user_input)

    if answer is None:
        st.session_state.awaiting_ticket = True
        st.session_state.conversation.append(("Assistant", "🤖 I'm not sure about that. You can submit a support ticket below for help from our team."))
    else:
        st.session_state.conversation.append(("Assistant", answer))
        st.session_state.awaiting_ticket = False

# Display conversation
for speaker, message in st.session_state.conversation:
    if speaker == "User":
        st.write(f"🧑‍💬 **You:** {message}")
    else:
        st.write(f"🤖 **Volkswagen Assistant:** {message}")

# Support ticket form
if st.session_state.awaiting_ticket:
    st.write("---")
    st.subheader("📩 Submit a Support Ticket")
    with st.form("ticket_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        summary = st.text_input("Summary of your issue")
        description = st.text_area("Describe your issue in detail")
        submitted = st.form_submit_button("Submit Ticket")
        if submitted:
            success = submit_support_ticket(name, email, summary, description)
            if success:
                st.success("✅ Your support ticket has been submitted! Our team will reach out to you soon.")
            else:
                st.error("❌ Failed to submit your ticket. Please try again.")

