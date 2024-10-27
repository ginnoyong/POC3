import streamlit as st
# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="PSSE AI Chatbot - Methodology"
)
# endregion <--------- Streamlit App Configuration --------->
st.title("Methodology")

c1=st.container(border=True)
with c1:
    st.subheader("Main Page")
    st.write("""
             The starting page of this app, where the user question is screened to ensure it's safe and appropriate.""") 
    st.write("""You can choose between two AI chat use cases using the buttons provided, \
                and your question will be directed to the appropriate AI chatbot.""")
    st.write("""Additionally, there is a Start Over Conversation button to reset the conversation. \
            If you notice that responses are not as expected, clearing the chat history can help improve performance.
            """)
    st.image("methodology\\PSSE-Methodology-PSSE_Chat.webp")

c2=st.container(border=True)
with c2:
    st.subheader("Use Case 1: Admission Exercises")
    st.write("""Upon launching the app, RAG (Retrieval-Augmented Generation) documents related to admission exercises \
             are loaded into the Admissions collection of the vector store.""")
    st.write("""One of the key objectives of this project is to ensure that the chatbot provides up-to-date answers. \
             To achieve this, the RecursiveUrlLoader package is used to gather information from the MOE Admissions \
             webpage and its child pages. \
             Additionally, a PDF document, directly sourced from an MOE URL, is also loaded into the same vector store collection \
             to enhance the chatbot’s accuracy.""")
    st.write("""Given that information about admission exercises is time-sensitive, the current date is inserted into the \
             user question before sending it to the LLM.""")
    st.write("""Additionally, the MultiQueryRetriever is utilized here to refine user prompts before passing them to \
             the Vector Store's retriever. This approach is found to produce better responses compared to using \
             MMR search type or a combination of both methods.""")

    st.image("methodology\\PSSE-Methodology-admissionadvisor.webp")

c3=st.container(border=True)
with c3:
    st.subheader("Use Case 2: POLITE Courses")
    st.write("""The initial approach was to load course info from the MOE Course Finder webpages using the RecursiveUrlLoader package also. \
             However, possibly due to the way the webpage loads its content (via API call), or other technical issues, this approach was unsuccessful.""")
    st.write("""An alternative method was attempted by loading info directly from individual Polytechnic and ITE websites, \
             but this led to an OpenAI RateLimitError, even when it is loading webpages from just one polytechnic.""")
    st.write("""To still achieve the goal of providing up-to-date information, an alternative approach was explored. \
             The WebResearchRetriever and GoogleSearchAPIWrapper packages from langchain_community are now being used to conduct on-demand \
             Internet searches via Google's Programmable Search Engine whenever a user query is received. \
             This search engine has been specifically configured to search only within the following sites: \
             www.moe.gov.sg/coursefinder/coursedetail?course=* and www.moe.gov.sg/coursefinder/*. """)
    st.write("""Additionally, the MultiQueryRetriever is also utilized here to refine user prompts before passing them to the WebResearchRetriever.""")
    st.write("""The CrewAI WebsiteSearchTool was also tested but yielded unsatisfactory results. \
             Its limited configuration options and lack of visibility into the underlying processes made it challenging to optimize.""")
    st.image("methodology\\PSSE-Methodology-courseadvisor.webp")