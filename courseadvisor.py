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
    chunk_size=200,
    chunk_overlap=50,
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
Note: \
    Post-Secondary Schools Education is made up of the Junior Colleges (a.k.a JC), Millennia Institutes (a.k.a MI), \
    Polytechnics (a.k.a Poly), and Institute of Technical Education (a.k.a ITE). \
    Do NOT include Universities in your answer.
    The words 'score' or 'points' in the user question is interchangable with 'Aggregate Score'. 
If necessary, use the chat history, delimited by <chat_history>, to help you understand \
    the question better. 
Always generate your answer based on the context provided, delimited by <context>. 
NEVER make up information. NEVER make up schools and courses that do not exist. 
If you don't know the answer, just say that you don't know.

Steps to follow to generate your response:
1. If the question is not about Post-Secondary School Education in Singapore, \
    explain why you are unable to provide any answers and provide an example what he/she can ask.
2. Analyse the question and identify the schools or courses in the context \
    that fit all the requirements in the question. \
3. Extract key information, such as School Name, Course Name, Course Code, Aggregate Score Range, Aggregate Score Type etc. \
    of these schools and courses from the context
4. Do NOT make up any info if they do not exist in the context. 
6. Use these steps to determine how good or likely will a student be accepted into a course or school \
    based on his/her aggregate score: \
        a. Identify the aggregate score range of the course / school.
        b. Let A be the smaller number in the aggregate score range, B be the bigger number.
        c. If a student's aggregate score is less than A, he/she has very good chance / is very likely to be accepted into the course. 
        d. If a student's aggregate score is between A and B, he/she has fair chance / is likely to be accepted into the course.
        e. If a student's aggregate score is more than B, he/she has poor chance / is unlikely to be accepted into the course.

Your answer should consist of a short abstract and \
the list of schools / courses that fulfill the question.
Generate the list of schools / courses in JSON format. \
Each JSON object will contain the key information of the schools/courses.
Omit JSON keys that are not applicable. Leave any unknown value blank.  

Sample of the JSON output of a list of schools or courses delimited by <json_list>: 
<json_list>
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
"Aggregate Score Range":"11-16",
"Aggregate Score Type":"ELR2B2-B"}},
]
</json_list>

Wrap the list of JSON objects delimited by <json_list> and </json_list> with NO other delimiters.

If the list of schools or courses in your answer contains Poly or ITE COURSES, \
include this link (MOE Course Finder)"https://www.moe.gov.sg/coursefinder" at the end of your answer.
If the list of schools or courses in your answer contains JC or MI SCHOOLS, \
include this link (MOE School Finder)"https://www.moe.gov.sg/schoolfinder" at the end of your answer.
Advise the user to use the link(s) to verify your answer. 

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
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",input_key="question", return_messages=True)

############ for testing
#memory.save_context({"question":"poly and ITE"}, {"output": "please elaborate"})
############

#~~~~~~~~ Improve human prompt for MultiQueryRetriever to find wider, related results.
from utility import convert_messages_to_llm_format
from llm import get_completion_by_messages

def improve_prompt(user_prompt):
    sys_prompt = """Your task is to improve a user question that will be fed into LangChain's MultiQueryRetriever. 
    Your improved prompt should be more concise and make the MultiQueryRetriever retrieve \
    more schools and courses that is relevant to the user question. 
    Respond with only the prompt."""
    #messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    messages = [{"role": "system", "content": sys_prompt},]
    #~~ inject chat history to improve the prompt
    formatted_messages = convert_messages_to_llm_format(memory.chat_memory.messages)
    messages.extend(formatted_messages)
    messages.extend([{"role": "user", "content": user_prompt}])
    print(messages)
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
from utility import process_courses_response

@st.cache_data
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

    #~~~~~~~~ split JSON from response text, convert it to df
    response_text, df_list = process_courses_response(response.get('result'))

    vectordb_courses.reset_collection()

    #return response.get('result')
    return response_text, df_list

def clear_memory():
   memory.clear()

def get_chat_history():
    #formatted_chat_messages = convert_messages_to_llm_format(memory.chat_memory.messages)
    #return pd.json_normalize(formatted_chat_messages)
    return memory.buffer

#~~~~~~~~~~~~~~~~ Testing code
#~~~~~~~~ Invoke and Response
#response = tool_vectordb_qachain_invoke("what are the JC's in singapore?")
#print(response.get('result'))
#~~~~~~~~ Invoke and Response
#user_prompt = "all cybersecurity courses in poly"
#user_prompt = improve_prompt("business courses")
#print(user_prompt)
#response = courses_invoke_question(user_prompt)
#formatted_chat_messages = convert_messages_to_llm_format(memory.chat_memory.messages)

#df_chat_history = pd.json_normalize(formatted_chat_messages)
#print(df_chat_history.to_string)
#print(response)
#~~~~~~~~ Invoke and Response
#response = tool_vectordb_qachain_invoke("tell me about the accountancy course in NYP?")
#print(response.get('result'))

#print(memory)

