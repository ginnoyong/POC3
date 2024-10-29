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
    chunk_size=500,
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
Your job is to answer the user question about Polytechnic (Poly) and Institute of Technical Education (ITE) courses in Singapore \
    that is presented at the end of this prompt.

Use the context, delimited by <context> to generate your answer. \
If you don't know the answer, just say that you don't know.
Do NOT make up information. Do NOT make up schools and courses that do not exist. 

Note:\
The words 'score' or 'points' in the user question refers to 'Aggregate Score'. 
The aggregate score range of a course indicate the scores of the students who were \
    accepted into the course in the previous admission exercise.
Better exam results gives a lower aggregate score. \
    a. Identify the aggregate score range of the course.
    b. Let A be the smaller number in the aggregate score range, B be the bigger number.
    c. If a student's aggregate score is less than A, he/she has very good chance / is very likely to be accepted into the course. 
    d. If a student's aggregate score is between A and B, he/she has fair chance / is likely to be accepted into the course.
    e. If a student's aggregate score is more than B, he/she has poor chance / is unlikely to be accepted into the course.

Follow these steps to generate your answer:
1. If the question is not about Poly and ITE courses in Singapore, \
    explain why you are unable to provide any answers and provide an example what he/she can ask.
2. Find courses and similar courses in the context that contains one or more of the key terms or similar in the user question.
3. Extract key information of these courses, such as \
    School Name, Course Name, Course Code, Aggregate Score Range, Aggregate Score Type etc. \
4. Formulate your answer to the user question based on these info that you find. 

If your answer contains a list of courses that answers the user question, \
generate the list in JSON format at the TOP of your answer. 
Each JSON object will contain the key information of the schools/courses.
Omit JSON keys that are not applicable. Leave any unknown value blank.
Sample of the JSON output of a list of courses delimited by <json_list>: 
<json_list>
[
{{"Institute Name":"Nanyang Polytechnic",
"School Name":"School of Business Management",
"Course Name":"Diploma in Sport and Wellness Management",
"Course Code":"C81",
"Aggregate Score Range":"11-16",
"Aggregate Score Type":"ELR2B2-B"}},
]
</json_list>
Wrap the list of JSON objects delimited by <json_list> and </json_list> with NO other delimiters.

Then followed by a concise text answer in response to the user question in your answer.
At the end provide the link to MOE Course Finder: https://www.moe.gov.sg/coursefinder \
and advise the user to verify the info or search for more courses. 

Add a line break at the end of your answer. 

<context>
{context}
</context>

<chat_history>
{chat_history}
</chat_history>
Question:{question}
Answer:"""

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
    Your improved prompt should make the MultiQueryRetriever retrieve \
        more courses that are related or similar to the core terms in the user question, \
            ultimately producing better results and improve the quality of the LLM's answer. 
    Respond with only the prompt."""
    #messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    messages = [{"role": "system", "content": sys_prompt},]
    #~~ inject chat history to improve the prompt
    formatted_messages = convert_messages_to_llm_format(memory.chat_memory.messages)
    messages.extend(formatted_messages)
    messages.extend([{"role": "user", "content": user_prompt}])
    #print(messages)
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

compressor = CohereRerank(top_n=12, model='rerank-english-v3.0')

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever_multiquery,
)

#~~~~~~~~ RetrievalQA code
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever_multiquery,
    #retriever=compression_retriever,
    #retriever=websearch_retriever,
    return_source_documents=False, # Make inspection of document possible
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, 
                       "memory":memory,
                       "verbose":False,
                       },
)

#~~~~ determine if another web retrieval is required
def determine_retrieval(user_prompt):
    sys_prompt = """Your task is to determine if the user question can be answered just by the information in the chat history.
        Respond with 'Y' if the user question CANNOT be answered by the information in the chat history alone. 
        Respond with 'N' if the user question CAN be answered by the information in the chat history alone.
        Respond with 'Y' if this is the first human input in the chat history.
    Respond with only a single letter 'Y' or 'N'."""
    #messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    messages = [{"role": "system", "content": sys_prompt},]
    #~~ inject chat history to improve the prompt
    formatted_messages = convert_messages_to_llm_format(memory.chat_memory.messages)
    messages.extend(formatted_messages)
    messages.extend([{"role": "user", "content": user_prompt}])
    #print(messages)
    response = get_completion_by_messages(messages = messages)
    print(response)
    return response

#~~~~ determine if another web retrieval is required
def respond_conversation(user_prompt):
    sys_prompt = """Your task is to answer the user question about Polytechnic and ITE education in Singapore based on the chat history. 
        Note:\
            The words 'score' or 'points' in the user question refers to 'Aggregate Score'. 
            The aggregate score range of a course indicate the scores of the students who were \
                accepted into the course in the previous admission exercise.
            A student with better academic results gets a lower aggregate score. \
                a. Identify the aggregate score range of the course.
                b. Let A be the smaller number in the aggregate score range, B be the bigger number.
                c. If a student's aggregate score is LESS than A, he/she has very good chance to be accepted into the course. 
                d. If a student's aggregate score is BETWEEEN A and B, he/she has fair chance to be accepted into the course.
                e. If a student's aggregate score is MORE than B, he/she has poor chance to be accepted into the course.
        """
    #messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    messages = [{"role": "system", "content": sys_prompt},]
    #~~ inject chat history to improve the prompt
    formatted_messages = convert_messages_to_llm_format(memory.chat_memory.messages)
    messages.extend(formatted_messages)
    messages.extend([{"role": "user", "content": user_prompt}])
    #print(messages)
    return get_completion_by_messages(messages = messages)

#~~~~ determine if question is valid
def check_question(user_prompt):
    sys_prompt = """Your task is to determine if the user question is about Polytechnic and ITE courses in Singapore, \
        or to asks for suggestions of Polytechnic and ITE courses in Singapore.
    Determine this in the context of the chat history also. 
    It can be assumed that the user question is likely to be about Polytechnic and ITE courses in Singapore, \
        so replace any ambiguous key terms accordingly.
        Respond with 'Y' if the user question is about Polytechnic and ITE courses in Singapore \
            or to asks for suggestions of Polytechnic and ITE courses in Singapore. 
        Respond with 'N' if the user question is NOT.
    Respond with only a single letter 'Y' or 'N'."""
    #messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    messages = [{"role": "system", "content": sys_prompt},]
    #~~ inject chat history to improve the prompt
    formatted_messages = convert_messages_to_llm_format(memory.chat_memory.messages)
    messages.extend(formatted_messages)
    messages.extend([{"role": "user", "content": user_prompt}])
    #print(messages)
    response = get_completion_by_messages(messages = messages)
    print(response)
    return response

#~~~~ invoke function to call from streamlit form
from logics import improve_message_courses
from utility import process_courses_response

@st.cache_data(ttl=300)
def courses_invoke_question(user_message):
    #result=search.run(user_message)
    #splitted_text = text_splitter(result)
    #vectordb_courses.from_texts(splitted_text)

    #~~~ determine if web search and RAG retrieval is required for this user question
    df_list = None
    if check_question(user_message)=='Y':
        if determine_retrieval(user_message)=='N':
            response_text = respond_conversation(user_message)
        else:
            user_message = improve_prompt(user_message)
            print(user_message)
            response = qa_chain.invoke(user_message)
            print(f"collection count:{vectordb_courses._collection.count()}")
            print(response.get('result'))
            if vectordb_courses._collection.count()==0:
                df_list = None
                response_text = f"""I am unable to retrieve any information that answers your question at this moment. \n\
                    Please try again later or search for specific courses at MOE Course Finder https://www.moe.gov.sg/coursefinder."""
            else:
                #~~~~~~~~ split JSON from response text, convert it to df
                response_text, df_list = process_courses_response(response.get('result'))
                    
            # reseting the vectordb_courses collection produces better responses. 
            vectordb_courses.reset_collection()
    else:
        response_text = "I can only answer questions about Polytechnic and ITE courses in Singapore. \
            You can ask questions such as 'What are the ELR2B2 score requirements for business courses in NYP?', or \
            'I like programming. What courses should I apply?', etc."
    #return response.get('result')
    return response_text, df_list

def clear_memory():
   memory.clear()

def get_chat_history():
    #formatted_chat_messages = convert_messages_to_llm_format(memory.chat_memory.messages)
    #return pd.json_normalize(formatted_chat_messages)
    return memory.buffer

def get_chat_history_msg():
   formatted_messages = convert_messages_to_llm_format(memory.chat_memory.messages)
   return formatted_messages

#~~~~~~~~~~~~~~~~ Testing code
#~~~~~~~~ Invoke and Response
#response = tool_vectordb_qachain_invoke("what are the JC's in singapore?")
#print(response.get('result'))
#~~~~~~~~ Invoke and Response
#user_prompt = "my interests are in food and beverage"
#user_prompt = improve_prompt(user_prompt)
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