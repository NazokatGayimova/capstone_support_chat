import streamlit as st
from qa_engine import ask_question, create_fake_support_ticket, get_company_info

st.set_page_config(page_title="Volkswagen AI Support", page_icon="🚗")

st.title("🚗 Volkswagen AI Customer Support")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your question here:")

if st.button("🔍 Search"):
    if user_input:
        result = ask_question(user_input)

        if result["answer"]:
            response = f"{result['answer']}\n\n📄 (Source: {result['citation']})\n{get_company_info()}"
        else:
            response = f"❗Sorry, I couldn't find information about that.\n\n{create_fake_support_ticket(user_input)}\n{get_company_info()}"

        st.session_state.chat_history.append((user_input, response))

if st.session_state.chat_history:
    st.subheader("🧵 Conversation History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")

