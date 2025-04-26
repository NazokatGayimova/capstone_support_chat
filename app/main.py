import streamlit as st
from qa_engine import ask_question, get_company_info, get_conversation_history

st.set_page_config(page_title="Volkswagen AI Customer Support", page_icon="🚗")
st.title("🚗 Volkswagen AI Customer Support")

company_info = get_company_info()

st.markdown(f"""
### 📞 Company Info:

**Name:** {company_info['name']}  
**📧 Email:** {company_info['email']}  
**📞 Phone:** {company_info['phone']}
""")

user_input = st.text_input("Type your question here:")

if st.button("Search"):
    if user_input:
        try:
            result = ask_question(user_input)
            st.markdown(f"### ✨ Answer:")
            st.markdown(f"{result['answer']}")

            if result["source"] and result["page"]:
                st.markdown(f"📄 (Source: {result['source']}, page {result['page']})")

            st.markdown("---")
            st.markdown(f"### 📞 Company Info:\n\nName: {company_info['name']} Email: {company_info['email']} Phone: {company_info['phone']}")

            if result["ticket_needed"]:
                st.warning("🤖 I couldn't find enough information. Please create a support ticket below.")
                with st.form("support_ticket"):
                    user_name = st.text_input("Your Name")
                    user_email = st.text_input("Your Email")
                    issue_summary = st.text_input("Issue Summary")
                    issue_description = st.text_area("Issue Description")
                    submitted = st.form_submit_button("Submit Support Ticket")
                    if submitted:
                        if user_name and user_email and issue_summary and issue_description:
                            st.success("✅ Support ticket submitted! Our team will contact you soon.")
                        else:
                            st.error("❌ Please fill in all fields to submit the ticket.")

            # 🧵 Show Conversation History
            conversation = get_conversation_history()
            if conversation:
                st.markdown("### 🧵 Conversation History:")
                for role, message in conversation:
                    st.markdown(f"**{role}:** {message}")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
    else:
        st.warning("Please type a question.")

