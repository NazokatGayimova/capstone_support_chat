import streamlit as st
from qa_engine import ask_question, submit_support_ticket, get_company_info

st.set_page_config(page_title="Volkswagen AI Customer Support", page_icon="🚗")

company_info = get_company_info()
with st.sidebar:
    st.header("📞 Company Info:")
    st.markdown(f"**Name:** {company_info['name']}")
    st.markdown(f"**📧 Email:** {company_info['email']}")
    st.markdown(f"**📞 Phone:** {company_info['phone']}")

st.title("🚗 Volkswagen AI Customer Support")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Type your question here:", key="user_input")

if st.button("🔍 Search"):
    if user_input:
        try:
            result = ask_question(user_input)

            # If the answer is short or uncertain, offer to submit ticket
            unclear_phrases = ["I'm not sure", "I don't know", "No information found", "cannot find", "not mentioned"]
            if any(phrase.lower() in result["answer"].lower() for phrase in unclear_phrases):
                st.warning("🤖 Sorry, I couldn't find an exact answer. Would you like to submit a support ticket?")
                if st.button("📨 Submit Support Ticket"):
                    name = st.text_input("Your Name", key="name")
                    email = st.text_input("Your Email", key="email")
                    issue_summary = st.text_input("Issue Summary", key="summary")
                    issue_description = st.text_area("Issue Description", key="description")
                    if st.button("✅ Confirm and Submit"):
                        submit_support_ticket(name, email, issue_summary, issue_description)
                        st.success("✅ Your support ticket has been created. Our team will reach out to you soon.")
            else:
                # Show answer and fake citation
                st.success(result["answer"])
                source_file, page_number = result["source"]
                st.info(f"📄 (Source: {source_file}, page {page_number})")

                # Save the conversation to history
                st.session_state.chat_history.append({
                    "user": user_input,
                    "assistant": result["answer"],
                    "source": f"{source_file}, page {page_number}"
                })

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# Display conversation history
if st.session_state.chat_history:
    st.divider()
    st.subheader("🧵 Conversation History")
    for i, chat in enumerate(st.session_state.chat_history, start=1):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['assistant']}")
        st.caption(f"📄 {chat['source']}")
