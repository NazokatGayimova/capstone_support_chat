import streamlit as st
from qa_engine import ask_question, get_company_info, get_conversation_history

st.set_page_config(page_title="Volkswagen AI Customer Support", page_icon="🚗", layout="wide")

# Header
st.title("🚗 Volkswagen AI Customer Support")

# Display company information
company_info = get_company_info()
st.markdown(f"""
### 📞 Company Info
**Name:** {company_info['name']}  
**📧 Email:** {company_info['email']}  
**📞 Phone:** {company_info['phone']}
""")

# User input
user_input = st.text_input("Type your question here:")

if st.button("Search"):
    if user_input.strip():
        try:
            # Get the response from the QA engine
            result = ask_question(user_input.strip())

            # Display the answer
            st.markdown("### ✨ Answer:")
            if result.get("answer"):
                st.markdown(result["answer"])

                # Display the source and page number if available
                if result.get("source") and result.get("page"):
                    st.markdown(f"📄 (Source: **{result['source']}**, Page **{result['page']}**)")
            else:
                st.warning("🤖 No relevant information found. You can submit a support ticket below.")

            # Support ticket form if no sufficient answer
            if result.get("ticket_needed"):
                st.warning("🤖 No sufficient information found. You can submit a support ticket below.")
                with st.form("support_ticket"):
                    user_name = st.text_input("Your Name")
                    user_email = st.text_input("Your Email")
                    issue_summary = st.text_input("Issue Summary")
                    issue_description = st.text_area("Issue Description")
                    submitted = st.form_submit_button("Submit Support Ticket")

                    if submitted:
                        # Basic form validation
                        if user_name.strip() and user_email.strip() and issue_summary.strip() and issue_description.strip():
                            # Placeholder for ticket submission logic
                            st.success("✅ Support ticket submitted! Our team will reach out to you soon.")
                        else:
                            st.error("❌ Please fill all fields and try again!")

            # Display conversation history
            st.markdown("### 🧵 Conversation History:")
            history = get_conversation_history()
            if history:
                for user, assistant in history:
                    st.markdown(f"**You:** {user}")
                    st.markdown(f"**Assistant:** {assistant}")
            else:
                st.info("No previous conversations found.")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
    else:
        st.warning("❌ Please type a question.")

# Footer for contact info
st.markdown("---")
st.markdown(f"""
### 📞 Company Info
**Name:** {company_info['name']}  
**📧 Email:** {company_info['email']}  
**📞 Phone:** {company_info['phone']}
""")

