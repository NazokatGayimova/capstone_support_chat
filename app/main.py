import streamlit as st
from qa_engine import ask_question, submit_support_ticket, get_company_info

st.set_page_config(page_title="Volkswagen AI Customer Support", page_icon="🚗")

st.title("🚗 Volkswagen AI Customer Support")

company_info = get_company_info()
st.markdown(f"**Company: {company_info['name']}**\n\n"
            f"📧 **Email:** {company_info['email']}\n\n"
            f"📞 **Phone:** {company_info['phone']}\n\n")

user_input = st.text_input("Type your question here:")

if user_input:
    with st.spinner("Searching for answer..."):
        try:
            answer, citation, found = ask_question(user_input)

            st.markdown(f"**Answer:** {answer}")

            if found and citation:
                st.markdown(f"📄 **Source:** {citation}")
            else:
                st.warning("🤖 I couldn't find a good answer. Would you like to submit a support ticket?")
                with st.form("support_ticket_form"):
                    name = st.text_input("Your Name")
                    email = st.text_input("Your Email")
                    summary = st.text_input("Ticket Summary")
                    description = st.text_area("Describe your issue")
                    submitted = st.form_submit_button("Submit Ticket")

                    if submitted:
                        if name and email and summary and description:
                            success = submit_support_ticket(name, email, summary, description)
                            if success:
                                st.success("✅ Your support ticket has been submitted! Our team will reach out to you soon.")
                        else:
                            st.error("⚠️ Please fill in all the fields before submitting.")

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

