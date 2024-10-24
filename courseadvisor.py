import os
import streamlit as st

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
#from langchain_google_community.search import GoogleSearchAPIWrapper
from langchain_community.utilities import GoogleSearchAPIWrapper
search = GoogleSearchAPIWrapper(k=10)



#~~~~~~~~ splitter: RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm import count_tokens
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=150,
    chunk_overlap=0,
    length_function=count_tokens
)


#~~~~~~~~ Chroma
from langchain_chroma import Chroma
vectordb_courses = Chroma(embedding_function=embeddings_model, collection_name='courses', persist_directory='./vector_db',)

#~~~~~~~~~ using WebResearchRetriever to fill the vectorstore and create retriever
from langchain_community.retrievers.web_research import WebResearchRetriever
websearch_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectordb_courses, llm=llm, search=search,
#    vectorstore=vectordb_courses, llm=llm, search=tool,
#    vectorstore=vectordb_courses, llm=llm_with_tools, search=search,
    allow_dangerous_requests=True,
    num_search_results=10,
    text_splitter=text_splitter,
)

#~~~~~~~~ Prompt Template code
from langchain.prompts import PromptTemplate

# Build prompt
template = """
<chat_history>
{chat_history}
</chat_history>

You are an expert in Post-Secondary School Education schools and courses in Singapore.
Your job is to list all the Post-Secondary Schools Education schools or courses in Singapore that \
    fulfill the requirements stated in the question. 
Post-Secondary Schools Education Schools are made up of the Junior Colleges (a.k.a JC), Millennia Institutes (a.k.a MI), \
    Polytechnics (a.k.a Poly), and Institute of Technical Education (a.k.a ITE).
If necessary, use the chat history, delimited by <chat_history>, to help you understand \
    the question better. 
Also generate your answer based on the context provided, delimited by <context>. 

Steps to follow to generate your response:
1. If the question is not about Post-Secondary School Education in Singapore, \
    remind the user what your job is and provide an example what he/she can ask.
2. Analyse the question and look for all the schools / courses that fulfill the requirements in the question 
4. The word 'score' in the question is likely to refer to 'Aggregate Score'.
5. Use these steps to determine how good / likely will a student be accepted into a course / school \
    based on his/her aggregate score: \
        a. Identify the aggregate score range of the course / school.
        b. Let A be the smaller number in the aggregate score range, B be the bigger number.
        c. If a student's aggregate score is less than A, he/she has very good chance / is very likely to be accepted into the course. 
        d. If a student's aggregate score is between A and B, he/she has fair chance / is likely to be accepted into the course.
        e. If a student's aggregate score is more than B, he/she has poor chance / is unlikely to be accepted into the course.

If you don't know the answer, just say that you don't know, NEVER make up answers. \
NEVER make up any information that do not exist. Leave the value blank if you do not know.

Your answer will be a list of JSON objects of the schools/courses that fulfill the question \
    each JSON object will contain key information of the schools/courses, such as: \
    School Name, Course Name, Course Code, Aggregate Score Range, Aggregate Score Type etc.
Omit JSON keys that are not applicable. 

Examples: \
[
{{"School Name":"Anglo-Chinese Junior College",
"Course Name":"Science",
"Course Code":"NA",
"Aggregate Score Range":"3-7",
"Aggregate Score Type":"L1R5"}},
{{"Institute Name":"Nanyang Polytechnic",
"School Name":"School of Business Management",
"Course Name":"Diploma in Sport and Wellness Management",
"Course Code":"C81",
"Aggregate Score Range":"11-16/7-9",
"Aggregate Score Type":"ELR2B2-B/ELMAB3"}},
]

Output the list of JSON objects delimited by <json_list> and </json_list> with NO other delimiters.
Add a line break at the end of your answer. 

<context>
{context}
</context>

Question:{question}
"""

#~~~~~~~~ the prompt template
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

#~~~~~~~~ ConversationBufferMemory code
from langchain.memory import ConversationSummaryMemory
# keep the same memory between both chat functions
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",input_key="question")

#~~~~~~~~ Improve human prompt for MultiQueryRetriever to find wider, related results.
from llm import get_completion_by_messages
def improve_prompt(user_prompt):
   sys_prompt = """Your task is to improve a user question that will be fed into LangChain's MultiQueryRetriever. 
   Your improved prompt should be more concise and make the MultiQueryRetriever retrieve \
    more schools and courses that is relevant to the user question. 
    Respond with only the prompt."""
   messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
   return get_completion_by_messages(messages = messages)

#~~~~~~~~ MultiQueryRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

# We will be reusing the `vectordb` from the Naive RAG
# You can imagine `MultiQueryRetriever` as a chain that generates multiple queries
# itself is not a complete RAG chain, but it can be used as a retriever in a RAG chain
retriever_multiquery = MultiQueryRetriever.from_llm(
  retriever=websearch_retriever, llm=llm, 
)

#~~~~~~~~ CohereRerank
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

compressor = CohereRerank(top_n=10, model='rerank-english-v3.0')

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever_multiquery,
)

#~~~~~~~~ RetrievalQA code
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    #retriever=websearch_retriever,
    return_source_documents=False, # Make inspection of document possible
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, 
                       "memory":memory,
                       "verbose":True,
                       },
)


#~~~~ invoke function to call from streamlit form
from logics import improve_message_courses
import pandas as pd
import json
import re

def courses_invoke_question(user_message):
    #result=search.run(user_message)
    #splitted_text = text_splitter(result)
    #vectordb_courses.from_texts(splitted_text)
    user_message = improve_prompt(user_message)
    print(user_message)
    response = qa_chain.invoke(user_message)
    print(vectordb_courses._collection.count())
    # find that reseting the vectordb_courses collection produces better responses. 
    print(response.get('result'))

    df_list=None
    # Use a regular expression to extract the JSON content within <json> tags
    json_strings = re.findall(r'<json_list>.+</json_list>', response.get('result'), flags = re.DOTALL)

    if len(json_strings)>0:
       json_strings = re.sub(r'</?json_list>','', json_strings[0])

    try:
        json_objs = json.loads(json_strings)
        df_list = pd.json_normalize(json_objs)
    except:
       pass

    response_text = re.sub(r'<json_list>.*</json_list>', '', response.get('result'), flags = re.DOTALL)

    vectordb_courses.reset_collection()

    #return response.get('result')
    return response_text, df_list

def clear_memory():
   memory.clear()

def get_chat_history():
   return memory.buffer
#~~~~~~~~~~~~~~~~ Testing code
#~~~~~~~~ Invoke and Response
#response = tool_vectordb_qachain_invoke("what are the JC's in singapore?")
#print(response.get('result'))
#~~~~~~~~ Invoke and Response
#user_prompt = "all cybersecurity courses in poly"
#user_prompt = improve_prompt("IT courses in poly and ITE")
#print(user_prompt)
#response = courses_invoke_question(user_prompt)
#print(response)
#~~~~~~~~ Invoke and Response
#response = tool_vectordb_qachain_invoke("tell me about the accountancy course in NYP?")
#print(response.get('result'))

#print(memory)

