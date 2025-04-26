import streamlit as st
from app.qa_engine import ask_question, get_company_info

st.set_page_config(page_title="Volkswagen AI Support", page_icon="🚗")
st.title("🚗 Volkswagen AI Customer Support")

company_info = get_company_info()

st.markdown(f"""
### 📞 Company Info:
- **Name**: {company_info['name']}
- **Email**: {company_info['email']}
- **Phone**: {company_info['phone']}
""")

# Input area
user_input = st.text_input("Type your question here:")

if st.button("Search") and user_input.strip() != "":
    with st.spinner("Searching..."):
        result = ask_question(user_input)

        if result["answer"]:
            st.success(result["answer"])

            if "citation" in result:
                st.markdown(f"""
                📄 **Source**: {result['citation']['file_name']}, page {result['citation']['page']}
                """)
        else:
            st.error("Sorry, we couldn't find the information you asked for. You can submit a support ticket!")
            with st.form("support_ticket"):
                name = st.text_input("Your Name")
                email = st.text_input("Your Email")
                summary = st.text_input("Ticket Summary")
                description = st.text_area("Ticket Description")
                submitted = st.form_submit_button("Submit Ticket")
                if submitted:
                    st.success("🎫 Support ticket created successfully! Our team will get back to you soon.")

