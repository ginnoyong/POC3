# Set up and run this Streamlit App
import streamlit as st
import llm # <--- This is the helper function that we have created 🆕

from admissionadvisor import admissions_invoke_question, clear_memory as cm_a
from courseadvisor import courses_invoke_question, clear_memory as cm_c
from utility import check_password  
from logics import check_malicious

# Check if the password is correct.  
if not check_password():  
    st.stop()

if "prompt_category" not in st.session_state:
    st.session_state.prompt_category = "Admission Exercises"

def select_admissions():
    st.session_state.prompt_category = "Admission Exercises"

def select_courses():
    st.session_state.prompt_category = "POLITE Courses"

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="PSSE AI Chatbot",
    
)
# endregion <--------- Streamlit App Configuration --------->

st.title("🎓 Post-Secondary School Education AI Chatbot")

#~~~ callback function to clear chat history
def clear_chat_history():
    match st.session_state.prompt_category:
        case "Admission Exercises":
            cm_a()
            st.session_state.messages = []
        case "POLITE Courses":
            cm_c()
            st.session_state.messages = []

df_list = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


ca = st.container(border=True)

for message in st.session_state.messages:
    #with st.chat_message(message["role"]):
    ca.chat_message(message["role"]).markdown(message["content"])

cb = st.container()
with cb:
    if user_prompt := st.chat_input(f"What questions do you have about {st.session_state.prompt_category}?"):
        st.session_state.messages.append({"role":"user","content":user_prompt})
        #with st.chat_message("user"):
        ca.chat_message("user").markdown(user_prompt)

        if check_malicious(user_prompt) == 'N':
            match st.session_state.prompt_category:
                case "Admission Exercises":
                    response = admissions_invoke_question(user_prompt)
                case "POLITE Courses":
                    response, df_list = courses_invoke_question(user_prompt)
                    #print(df_list.to_string())
                case _:
                    response = "Please select one of the question categories to begin."
        else:
            response = "Unable to process this question."

        st.session_state.messages.append({"role":"assistant","content":response})
        #with st.chat_message("assistant"):
        ca.chat_message("assistant").markdown(response)
    #       c7.chat_message("assistant").markdown(response)

if df_list is not None:
    st.dataframe(df_list)
            
        
with st.expander("🚨IMPORTANT NOTICE"):
    st.write(f'''
        This web application is a prototype developed for educational purposes only. \
            The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, \
            especially those related to financial, legal, or healthcare matters.''')
    st.write(f'''Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. \
            You assume full responsibility for how you use any generated output. \
            Always consult with qualified professionals for accurate and personalized advice.''')
        

c1 = st.sidebar.container(border=True)
with c1:
    st.subheader(f"Ask questions about:")
    col1, col2 = st.columns(2)
    with col1:
        st.button("Admission Exercises", on_click=select_admissions)
    with col2:
        st.button("POLITE Courses", on_click=select_courses)

#~~~ button to clear chat history
st.sidebar.button("Start Over Conversation", on_click=clear_chat_history)
