import streamlit as st
import random  
import hmac  
import pandas as pd
import re
import json

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


#~~~~~~~~ split JSON from response text, convert it to df
def process_courses_response(response_text):
    df_list=None
    # Use a regular expression to extract the JSON content within <json> tags
    json_strings = re.findall(r'<json_list>.+</json_list>', response_text, flags = re.DOTALL)

    if len(json_strings)>0:
       json_strings = re.sub(r'</?json_list>','', json_strings[0])

    try:
        json_objs = json.loads(json_strings)
        df_list = pd.json_normalize(json_objs)
    except:
       pass

    response_text = re.sub(r'<json_list>.*</json_list>', '', response_text, flags = re.DOTALL)

    return response_text, df_list