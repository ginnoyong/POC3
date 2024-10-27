# Set up and run this Streamlit App
import streamlit as st
import llm # <--- This is the helper function that we have created 🆕

from admissionadvisor import admissions_invoke_question, clear_memory as cm_a, get_chat_history as gch_a
from courseadvisor import courses_invoke_question, clear_memory as cm_c, get_chat_history as gch_c
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
    page_title="PSSE AI Chatbot"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("🎓 Post-Secondary School Education AI Chatbot")
c1 = st.container(border=True)
with c1:
    st.subheader(f"Ask questions about {st.session_state.prompt_category}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Admission Exercises", on_click=select_admissions)
    with col2:
        st.button("POLITE Courses", on_click=select_courses)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

#~~~ callback function to clear chat history
def clear_chat_history():
    match st.session_state.prompt_category:
        case "Admission Exercises":
            cm_a()
            st.session_state.chat_history = None
        case "POLITE Courses":
            cm_c()
            st.session_state.chat_history = None

#~~~ callback function to update chat history after every submit
def get_chat_history():
    match st.session_state.prompt_category:
        case "Admission Exercises":
            st.session_state.chat_history = gch_a()
        case "POLITE Courses":
            st.session_state.chat_history = gch_c()

df_list = None

form = st.form(key="form")
user_prompt = form.text_area(f"What questions do you have about {st.session_state.prompt_category}?", height=200)
if form.form_submit_button("Submit"):
    st.toast(f"User Input Submitted - {user_prompt}")
    #response = llm.get_completion(user_prompt) # <--- This calls the helper function that we have created 🆕
    #response = admissions_invoke_question(user_prompt)
    #prompt_category = categorise_prompt(user_prompt)
    #response = courses_invoke_question(user_prompt)
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

    print(f"\n\nUser Input is {user_prompt}")

    if df_list is not None:
        st.dataframe(df_list)
    
    st.write(response) # <--- This displays the response generated by the LLM onto the frontend 🆕
        
        #~~~ update chat history after each submit
    get_chat_history()

c3 = st.container(border=True)
with c3:
    with st.expander(f"Conversation Summary on {st.session_state.prompt_category}"):
        st.write(f'''
            {st.session_state.chat_history}
        ''')
        #st.dataframe(st.session_state.chat_history)

c4 = st.container(border=True)
with c4:
    with st.expander(f"""IMPORTANT NOTICE"""):
        st.write(f'''
            This web application is a prototype developed for educational purposes only. \
                The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, \
                especially those related to financial, legal, or healthcare matters.''')
        st.write(f'''Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. \
                 You assume full responsibility for how you use any generated output. \
                 Always consult with qualified professionals for accurate and personalized advice.''')

#~~~ button to clear chat history
st.sidebar.button("Start Over Conversation", on_click=clear_chat_history)
