import streamlit as st


class ChatMemory:

    @staticmethod
    def initialize():

        if "messages" not in st.session_state:
            st.session_state.messages = []

    @staticmethod
    def add(role, content):

        st.session_state.messages.append({
            "role": role,
            "content": content
        })

    @staticmethod
    def get():

        return st.session_state.messages

    @staticmethod
    def clear():

        st.session_state.messages = []