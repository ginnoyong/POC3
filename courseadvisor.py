import os
import streamlit as st
from langchain_community.document_loaders import RecursiveUrlLoader
import re
from bs4 import BeautifulSoup

try:
    #~~~ only when deploying
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    #~~~ only when deploying
except:
    pass

from dotenv import load_dotenv
if load_dotenv('.env'):
   OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
else:
   OPENAI_API_KEY=st.secrets['OPENAI_API_KEY']
   
#~~~~~~~~ Embeddings code
from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

#~~~~~~~~ ChatOpenAI llm
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

#~~~~~~~~~ using the deprecated langchain_community.utilities version cos it works with WebResearchRetriever
#~~~~~ unable to use RecursiveUrlLoader to crawl moe course finder website, think data is behind a dynamic api call
#~~~~~ unable to use RecursiveUrlLoader to crawl individual polytechnic websites, too many pages, got rate limit error trying to do embedding
#~~~~~ resort to using google api search to augment the context info
# the GOOGLE_API_KEY and GOOGLE_CSE_ID keys are essential here
#from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.utilities import GoogleSearchAPIWrapper
search = GoogleSearchAPIWrapper()

#~~~~~~~~~ using WebResearchRetriever to fill the vectorstore and create retriever
from langchain_community.retrievers.web_research import WebResearchRetriever

from langchain_chroma import Chroma
vectordb_courses = Chroma(embedding_function=embeddings_model, collection_name='courses', persist_directory='./vector_db')
retriever = WebResearchRetriever.from_llm(
    vectorstore=vectordb_courses, llm=llm, search=search,
#    vectorstore=vectordb_courses, llm=llm_with_tools, search=search,
    allow_dangerous_requests=True,
    num_search_results=8,
)

#~~~~~~~~ Prompt Template code
from langchain.prompts import PromptTemplate

# Build prompt
template = """
<chat_history>
{chat_history}
</chat_history>

You are an expert in Post-Secondary School Education schools and courses in Singapore.
Your scope of schools and courses are the Junior Colleges (a.k.a JC), Millennia Institutes (a.k.a MI), \
    Polytechnics (a.k.a Poly), and Institute of Technical Education (a.k.a ITE).
The prompts could be in Singlish. Singaporeans like to use abbreviation a lot. 

Use the chat history, delimited by <chat_history>, and context information, delimited by <context>, \
    to answer the question at the end. 

Steps to follow to generate your response:
1. If the question is not about Post-Secondary School Education in Singapore, \
    remind the user what your job is and provide an example what he/she can ask.
2. Analyse the question and look for the relevant schools / courses provided in the context information. 
3. You must extract these information of the schools / courses that you use in your answer: \
    a. Name of the JC, MI, Poly and/or ITE
    b. Course name and course code (if applicable)
    c. Aggregate score range, for example, 6 to 10, or 6 - 10, etc.
    d. Type of Aggregate score, for example, ELR2B2-A, L1R5, ELMAB3, etc.
    e. 'Score' in the question is likely to refer to 'Aggregate Score'.
4. Use this method to determine how good / likely will a student be accepted into a course / school \
    based on his/her aggregate score: \
        a. Identify the aggregate score range of the course / school.
        b. Let A be the smaller number in the aggregate score range, B be the bigger number.
        c. If a student's aggregate score is less than A, he/she has very good chance / is very likely to be accepted into the course. 
        d. If a student's aggregate score is between A and B, he/she has fair chance / is likely to be accepted into the course.
        e. If a student's aggregate score is more than B, he/she has poor chance / is unlikely to be accepted into the course.

If you don't know the answer, just say that you don't know, NEVER make up answers. \
NEVER make up schools / courses that do not exist.

Be polite. Keep the answer comprehensive and  concise. 
Always add "For more informatio
Add a line break at the end of your answer. 

Think about what the user might want to ask about next \
    and suggest with 'Would you also like to find out...' after that.

<context>
{context}
</context>

Question:{question}
Helpful Answer:
Would you also like to find out:
"""

#~~~~~~~~ the prompt template
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

#~~~~~~~~ ConversationBufferMemory code
from langchain.memory import ConversationSummaryMemory
# keep the same memory between both chat functions
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",input_key="question")

#~~~~~~~~ RetrievalQA code
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    #retriever=web_search_retriever,
    retriever=retriever,
    return_source_documents=True, # Make inspection of document possible
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, 
                       "memory":memory,
                       "verbose":True},
)


#~~~~ invoke function to call from streamlit form
from logics import improve_message_courses

def courses_invoke_question(user_message):
    #result=search.run(user_message)
    #splitted_text = text_splitter(result)
    #vectordb_courses.from_texts(splitted_text)
    #user_message = improve_message_courses(user_message)
    response = qa_chain.invoke(user_message)
    # find that reseting the vectordb_courses collection produces better responses. 
    vectordb_courses.reset_collection()
    return response.get('result'), memory.buffer

def clear_memory():
   memory.clear()
#~~~~~~~~~~~~~~~~ Testing code
#~~~~~~~~ Invoke and Response
#response = tool_vectordb_qachain_invoke("what are the JC's in singapore?")
#print(response.get('result'))
#~~~~~~~~ Invoke and Response
#response = tool_vectordb_qachain_invoke("List all accountancy-related courses in all the polytechnics.")
#print(response.get('result'))
#~~~~~~~~ Invoke and Response
#response = tool_vectordb_qachain_invoke("tell me about the accountancy course in NYP?")
#print(response.get('result'))

#print(memory)

