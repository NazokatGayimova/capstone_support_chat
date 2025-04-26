import streamlit as st
from qa_engine import ask_question, submit_support_ticket, get_company_info

st.set_page_config(page_title="Volkswagen AI Customer Support 🚗")

company_info = get_company_info()

st.title("🚗 Volkswagen AI Customer Support")
st.markdown(f"**Company:** {company_info['name']}")
st.markdown(f"📧 **Email:** {company_info['email']}")
st.markdown(f"📞 **Phone:** {company_info['phone']}")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("Type your question here:")

if st.button("🔎 Search"):
    if user_input:
        try:
            answer, citation, found = ask_question(user_input)
            if found:
                response = f"{answer}\n\n📄 (Source: {citation})"
            else:
                response = (
                    "🤖 I'm not sure based on our documents. "
                    "Would you like to submit a support ticket?"
                )

            st.session_state.conversation.append(("You", user_input))
            st.session_state.conversation.append(("Assistant", response))

            if not found:
                with st.form(key="support_ticket_form"):
                    name = st.text_input("Your Name")
                    email = st.text_input("Your Email")
                    if st.form_submit_button("📩 Submit Support Ticket"):
                        ticket_info = submit_support_ticket(name, email, user_input)
                        st.success(ticket_info)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

st.markdown("### 🧵 Conversation History")
for speaker, text in st.session_state.conversation:
    st.markdown(f"**{speaker}:** {text}")

st.markdown("---")
st.markdown(f"📞 **Company Info:**\n\nName: {company_info['name']}  \nEmail: {company_info['email']}  \nPhone: {company_info['phone']}")

