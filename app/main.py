import streamlit as st
from qa_engine import ask_question, submit_support_ticket, get_company_info

st.set_page_config(page_title="Volkswagen AI Customer Support 🚗")

company_info = get_company_info()

st.title("🚗 Volkswagen AI Customer Support")
st.markdown(f"**Company:** {company_info['name']}")
st.markdown(f"📧 **Email:** {company_info['email']}")
st.markdown(f"📞 **Phone:** {company_info['phone']}")

user_input = st.text_input("Type your question here:")

if st.button("🔎 Search"):
    if user_input:
        try:
            answer, citation, found = ask_question(user_input)

            if found and citation:
                final_answer = f"**Answer:** {answer}\n\n📄 **Source:** {citation}"
            else:
                final_answer = (
                    "🤖 I'm not sure based on the available documents.\n\n"
                    "Would you like to submit a support ticket below?"
                )

            st.markdown("## 🧵 Latest Answer")
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**Assistant:** {final_answer}")

            if not found:
                with st.form(key="support_ticket_form"):
                    name = st.text_input("Your Name")
                    email = st.text_input("Your Email")
                    if st.form_submit_button("📩 Submit Support Ticket"):
                        ticket_info = submit_support_ticket(name, email, user_input)
                        st.success(ticket_info)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

