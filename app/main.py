import streamlit as st
from qa_engine import ask_question

st.set_page_config(page_title="Volkswagen AI Support", page_icon="🚗")

st.title("🚗 Volkswagen AI Customer Support")

st.divider()

# Company Info
st.markdown(f"""
📞 **Company Info:**

Name: Volkswagen Group  
📧 Email: support@volkswagen.com  
📞 Phone: +49 5361 9000
""")

st.divider()

# User input
user_input = st.text_input("Type your question here:")

# Search button
if st.button("🔎 Search") and user_input:
    with st.spinner("Searching for the best answer..."):
        result = ask_question(user_input)

    if result["type"] == "answer":
        st.success(result["answer"])

        st.markdown(f"📄 **Source:** `{result['citation_file']}`, page {result['citation_page']}")
        st.markdown(f"""
📞 **Company Info:**

Name: {result['company_info']['name']}  
📧 Email: {result['company_info']['email']}  
📞 Phone: {result['company_info']['phone']}
""")

    elif result["type"] == "support_ticket":
        st.warning(result["message"])
        with st.form("support_ticket_form"):
            user_name = st.text_input("Your Name")
            user_email = st.text_input("Your Email")
            issue_summary = st.text_input("Issue Summary")
            issue_description = st.text_area("Issue Description")

            submitted = st.form_submit_button("Submit Support Ticket")
            if submitted:
                st.success(f"✅ Thank you, {user_name}! Your support ticket has been submitted. Our team will reach out to {user_email} soon.")

    elif result["type"] == "error":
        st.error(f"❌ An error occurred: {result['error_message']}")

# Footer
st.divider()
st.caption("Volkswagen AI Support Chat © 2025")

