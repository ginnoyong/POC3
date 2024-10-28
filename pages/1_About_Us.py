import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="PSSE AI Chatbot - About Us"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("About Us")
c3 = st.container(border=True)
with c3:
    st.subheader("Background")
    st.write("""As an academic staff member at a Polytechnic in Singapore, my responsibilities extend beyond teaching. \
            One of the less enjoyable tasks is providing guidance during major admission exercises. The admission process can be complex, \
            and academic staff can be less than being conversant in its knowledge as it is not part of our core function.""") 
    st.write("""With approximately 30,000 students enrolling in Polytechnics and ITE each year, there is a high demand for assistance from \
            secondary school leavers and their parents or guardians. During admission periods, the volume of inquiries can be overwhelming for \
            only the admission staff to manage alone.""")
    st.write("""Additionally, some inquiries are course-specific, which can be challenging for the admission staff. \
            It can also be challenging even for the academic staff, if the question pertains to a course outside of their own department.""")
    st.write("""This chatbot was designed to address these challenges. By providing accurate and up-to-date information, \
            it aims to help prospects navigate the admission process more smoothly and efficiently. \
            It also helps to reduces the amount of preparation required, ultimately saving time for both admission and academic staff involved in advising duties.""")
    st.write("""This chatbot could also be used on ad-hoc basis by both students and staff for education and career guidance \
             and progression planning purposes.""")

c1 = st.container(border=True)
with c1:
    st.subheader("Project Scope")
    st.write("""
    The project involves developing an AI-powered chatbot designed to address two key use cases in Post-Secondary School Education: \
             **Admission Exercises** and **Polytechnic and ITE Courses**. \
             The chatbot will cater to not only prospective students and their parents or guardians, but also both academic and admission staff \
             working in Polytechnics and ITE. It aims to offer accurate, up-to-date information and guidance across these two areas.
    """)

    # Display the first use case
    st.write("**1. Admission Exercises**")
    st.write("""
    - The chatbot will assist users by providing detailed information on Post-Secondary School Education Admission Exercises. \
            This includes application deadlines, eligibility criteria, required documents, and step-by-step guidance on how to apply.
    - It will be capable of answering general questions about the admission procedures, helping users understand important timelines and requirements \
             , ensuring that users receive accurate and timely information.
    - Examples of question you can ask, \
             'What admission exercises are for Polytechnic intake?', \
             followed by 'For N-level student?'
    """)

    # Display the second use case
    st.write("**2. Polytechnic and ITE Courses**")
    st.write("""
    - The chatbot will provide key information on specific courses offered across different Polytechnics and ITEs, including course name and code, \
             synopses, career prospects, entry aggregate score range, and more.
    - Users can ask about particular courses, or general field of studies. \
             The chatbot will deliver tailored responses enabling prospective students to make informed decisions.
    - By integrating with official MOE website, \
             the chatbot ensures that the information provided is current and accurate, \
             helping users navigate their options effectively.
    - Examples of question you can ask, \
             'what accountancy courses are there in polytechnics?', \
             followed by 'If my elr2b2 score is 12, which of these should I apply?'
    """)

c2 = st.container(border=True)
with c2:
    st.subheader("Objectives")
    st.write("**1. Accurate and Timely Information**")
    st.write("Deliver up-to-date answers on Post-Secondary School Education admission exercises, and course information across Polytechnics and ITEs.")

    # Objective 2
    st.write("**2. Enhance User Experience**")
    st.write("Provide quick, accurate information to reduce the need for face-to-face consultations during peak admission exercise periods.")

    # Objective 3
    st.write("**3. Reduce Staff Workload**")
    st.write("""
            - Automate responses to queries on complex admission processes. 
            - Provide details of vast variety of courses across different institutions and schools in response to student's enquiry. 
            - Reducing the resource demand on academic and admission staff to prepare for knowledge that are not in their function domains.
            """)

    # Objective 4
    st.write("**4. Seamless Data Integration**")
    st.write("Integrate with official and live data sources to provide up-to-date information.")

c4 = st.container(border=True)
with c4:
    st.subheader("Data Sources")
    st.write("""
        **Use Case 1: Admission Exercises**
        - MOE Post-Secondary Admissions exercises and programmes: https://www.moe.gov.sg/post-secondary/admissions
        - MOE Post-Secondary Admissions Exercises booklet: https://www.moe.gov.sg/-/media/files/post-secondary/a-guide-to-post-secondary-admissions-exercises.pdf
        
        **Use Case 2: POLITE Courses**
        - MOE Course Finder: https://www.moe.gov.sg/coursefinder
             """)
    st.write("""
             **Data Classification**
             - Open
             """)
    
c5 = st.container(border=True)
with c5:
    st.subheader("Features")
    st.write("""
        **1. Live RAG sources**
        - In order to achieve the objective of always up-to-date information, \
             RAG documents are loaded directly from webpages or documents hosted on official websites
        - Use Case 1: RAG documents are loaded directly from webpages \
             and a PDF document hosted on MOE website (https://www.moe.gov.sg/post-secondary/admissions, \
             https://www.moe.gov.sg/-/media/files/post-secondary/a-guide-to-post-secondary-admissions-exercises.pdf).  
        - Use Case 2: RAG documents are loaded on-demand by performing Google searches \
             in specified MOE webpages when the user question is received \
             (https://www.moe.gov.sg/coursefinder/*, https://www.moe.gov.sg/coursefinder/coursedetail?course=*). \
             See Methodology on why a different RAG loading method is used. 
        
        **2. Conversation-Style Chat**
        - Chat history is incorported at key stages of the chatbot's process, including to refine user queries, \
             generating responses, and deciding if additional RAG (Retrieval-Augmented Generation) retrieval is necessary. 
        - This ensures more context-aware and accurate answers.
        
        **3. Viewing and Clearing Conversation Summary**
        - Users can view a summary of the current chat history for easy reference.
        - The conversation summary can be cleared via a dedicated button, effectively resetting the chat.
        
        **4. Downloadable Course Details**
        - Course details provided in the chatbot's responses are converted into a downloadable data frame.
        - This feature allows users to easily compile and compare course information from multiple chatbot interactions for quick reference.
             
    """)
    
