import streamlit as st
from qa_engine import ask_question, get_company_info

st.set_page_config(page_title="Volkswagen AI Customer Support", page_icon="🚗")

company = get_company_info()

st.title("🚗 Volkswagen AI Customer Support")
st.markdown(f"**Company: {company['name']}**\n\n📧 Email: {company['email']}\n\n📞 Phone: {company['phone']}")

user_input = st.text_input("Type your question here:")

if st.button("Search"):
    if user_input:
        with st.spinner("Searching for the best answer..."):
            try:
                answer = ask_question(user_input)
                st.success(answer)
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

