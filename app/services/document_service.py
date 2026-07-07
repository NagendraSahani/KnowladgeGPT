import streamlit as st


class DocumentService:

    @staticmethod
    def save_document(text):

        st.session_state["document_text"] = text

    @staticmethod
    def get_document():

        return st.session_state.get(
            "document_text",
            ""
        )