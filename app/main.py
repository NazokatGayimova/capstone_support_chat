import streamlit as st
from qa_engine import ask_question, submit_support_ticket, get_company_info

st.set_page_config(page_title="Volkswagen AI Customer Support", page_icon="🚗")

st.title("🚗 Volkswagen AI Customer Support")

company_info = get_company_info()
st.markdown(f"**Company: {company_info['name']}**\n\n"
            f"📧 **Email:** {company_info['email']}\n\n"
            f"📞 **Phone:** {company_info['phone']}")

st.write("---")

# 🔵 Create a text_input inside a form (this way, SEARCH BUTTON appears!)
with st.form(key="question_form"):
    user_question = st.text_input("Type your question here:")
    submit_button = st.form_submit_button(label="Ask")

if submit_button:
    if user_question:
        with st.spinner("Searching for the answer..."):
            try:
                answer, citation, found = ask_question(user_question)

                st.markdown(f"**Answer:** {answer}")

                if found and citation:
                    st.markdown(f"📄 **Source:** {citation}")
                else:
                    st.warning("🤖 I couldn't find the answer. Please submit a support ticket:")
                    with st.form("support_ticket_form"):
                        name = st.text_input("Your Name")
                        email = st.text_input("Your Email")
                        summary = st.text_input("Ticket Summary")
                        description = st.text_area("Describe your issue")
                        ticket_submitted = st.form_submit_button("Submit Ticket")

                        if ticket_submitted:
                            if name and email and summary and description:
                                success = submit_support_ticket(name, email, summary, description)
                                if success:
                                    st.success("✅ Support ticket submitted successfully! Our team will get back to you soon.")
                            else:
                                st.error("⚠️ Please fill in all fields before submitting.")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please type a question before submitting.")

