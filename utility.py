import streamlit as st
import random  
import hmac  

def check_password():  
    """Returns `True` if the user had the correct password."""  
    def password_entered():  
        """Checks whether a password entered by the user is correct."""  
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):  
            st.session_state["password_correct"] = True  
            del st.session_state["password"]  # Don't store the password.  
        else:  
            st.session_state["password_correct"] = False  
    # Return True if the passward is validated.  
    if st.session_state.get("password_correct", False):  
        return True  
    # Show input for password.  
    st.text_input(  
        "Password", type="password", on_change=password_entered, key="password"  
    )  
    if "password_correct" in st.session_state:  
        st.error("😕 Password incorrect")  
    return False

#~~~~~~~~ Convert HumanMessage, AIMessage to LLM format messages
from langchain_core.messages import HumanMessage, AIMessage

def convert_messages_to_llm_format(memory_messages):
    formatted_messages = []
    for message in memory_messages:
        if isinstance(message, HumanMessage):
            formatted_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            formatted_messages.append({"role": "assistant", "content": message.content})
    return formatted_messages