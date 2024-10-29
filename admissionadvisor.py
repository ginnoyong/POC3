import os
import streamlit as st
from dotenv import load_dotenv

try: 
    #~~~ only when deploying
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    #~~~ only when deploying
except:
   pass

if load_dotenv('.env'):
   OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
else:
   OPENAI_API_KEY=st.secrets['OPENAI_API_KEY']

#~~~~~~~~ RecursiveUrlLoader Code
from langchain_community.document_loaders import RecursiveUrlLoader
import re
from bs4 import BeautifulSoup

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, features="html.parser")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

loader = RecursiveUrlLoader(
    "https://www.moe.gov.sg/post-secondary/admissions",
    #"https://docs.python.org/3.9/",
    #~~~ do not use bs4_extractor if using HTML splitters
    #extractor=bs4_extractor,
    max_depth=3,
    # use_async=False,
    # metadata_extractor=None,
    # exclude_dirs=(),
    # timeout=10,
    # check_response_status=True,
    # continue_on_failure=True,
    # prevent_outside=True,
    # base_url=None,
    # ...
)

docs = loader.load()
#print(f"""\n\n#####\n{docs[0].metadata}\n{docs[0].page_content}\n""")
#print(f"""\n\n#####\n{docs[1].metadata}\n{docs[1].page_content}\n""")

#~~~~~~~~ HTMLSectionSplitter
#~~~~ this code is to resolve incompatibility issue
#~~~~ RecursiveUrlLoader produces metadata['title'] but HTMLSectionSplitter.split_documents expects metadata['Title'] 
for doc in docs:
   doc.metadata['Title']=doc.metadata.get('title')
#print(docs[0].metadata)


#~~~~~~~~ Embeddings code
from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

#~~~~~~~~ ChatOpenAI llm
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)


#~~~~~~~~ Select type of splitters to use, remove comment tags to use
#~~~~~~~~ splitter: HTMLSectionSplitter
from langchain_text_splitters import HTMLSectionSplitter
headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ]
html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
#~~~ comment out if using ParentDocumentRetriever
html_splitted_docs = html_splitter.split_documents(documents=(docs))
#print(len(html_splitted_docs))


#~~~~~~~~~~~~~~~~ Combine both Website and PDF documents for better RAG results. 
#~~~~~~~~ splitter: SemanticChunker 
from langchain_experimental.text_splitter import SemanticChunker

# Create the text splitter
semantic_text_splitter = SemanticChunker(embeddings_model)

#~~~~~~~~ splitter: RecursiveCharacterTextSplitter
#from langchain_text_splitters import RecursiveCharacterTextSplitter
#text_splitter = RecursiveCharacterTextSplitter(
#    separators=["\n\n", "\n", " ", ""],
#    chunk_size=500,
    # chunk_overlap=50,
    # length_function=count_tokens
#)


#~~~~~~~~ load pdf doc: PyPDFLoader
file_path = "https://www.moe.gov.sg/-/media/files/post-secondary/a-guide-to-post-secondary-admissions-exercises.pdf"

from langchain_community.document_loaders import PyPDFLoader
 
loader = PyPDFLoader(file_path)
pdf_docs = loader.load()

splitted_pdf_docs = semantic_text_splitter.split_documents(pdf_docs)

#~~~~~~~~
combined_splitted_docs = []
combined_splitted_docs.extend(splitted_pdf_docs)
combined_splitted_docs.extend(html_splitted_docs)

#~~~~~~~~ Chroma Vector Store code
from langchain_chroma import Chroma

#~~~ comment out if using ParentDocumentRetriever
vectordb = Chroma.from_documents(combined_splitted_docs, embeddings_model, collection_name='admissions', persist_directory='./vector_db')
# vectordb.reset_collection()
# print(f"########:{len(vectordb.get()['documents'])}")

#~~~~~~~~ ParentDocumentRetriever code
#from langchain.storage import InMemoryStore
#from langchain.retrievers import ParentDocumentRetriever
#from langchain_text_splitters import RecursiveCharacterTextSplitter

#child_splitter = RecursiveCharacterTextSplitter.from_language('html')
#print(RecursiveCharacterTextSplitter.get_separators_for_language('html'))
#store = InMemoryStore()
#vectordb = Chroma(embedding_function=embeddings_model, collection_name='admissions', persist_directory='./vector_db')
#retriever = ParentDocumentRetriever(
#    vectorstore=vectordb,
#    docstore=store,
#    child_splitter=child_splitter,
#    search_kwargs={"fetch_k": 20,"k":5}, 
#    search_type="mmr",
#)

#retriever.add_documents(docs, ids=None)

#~~~~~~~~ ConversationBufferMemory code
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
# memory = ConversationBufferMemory(memory_key="chat_history",input_key="question")
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",input_key="question", return_messages=True)

#~~~~~~~~ Prompt Template code
from langchain.prompts import PromptTemplate


# Build prompt
template = """
You are an expert in Post-Secondary School Education admission matters in Singapore.
Your job is to provide answer about Post-Secondary School Education admission exercises \
to the user's question at the end.

1. If the question is not about Post-Secondary School Education admission matters in Singapore, \
    explain why you are not able to provide an answer and provide an example what he/she can ask.  
2. Find your answer to the user's question in the context information, delimited by <context>.
3. If you don't know the answer, just say that you don't know, do NOT make up answers. \
4. Do NOT make up information that do not exist in the context.
5. Pay attention to the dates and timelines of admission exercises in the context info. \
    Craft your answer with these dates and timelines in perspective. \
    Provide known dates available in the context where applicable. 

Keep the answer as concise as possible.

Provide this link (MOE Post-Secondary School Admissions)"https://www.moe.gov.sg/post-secondary/admissions" \
at the end of your answer and advise the user to use it to \
verify your answer. 

Add a line break after your answer.

Think about what the user might want to ask about next \
    and suggest with 'Would you also like to find out...' after that.
    Restrict suggestions only to things you know, i.e. about post-secondary school education admissions.

<context>
Different admission exercises are meant specifically for different types of students. 
There are 2 main types of Secondary School students in Singapore: \
O-Level and N-Level. N-Level is made up of N(A), stands for Normal Academic, \
    and N(T), stands for N(T).
Direct-Entry-Scheme to Polytechnic Programme (DPP) and Polytechnic Foundation Programme (PFP) \
are also Admission Exercises.
{context}
</context>

{chat_history}
Question:{question}
Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
#QA_CHAIN_PROMPT = PromptTemplate(
#    template=template,
#    input_variables=["chat_history", "context", "question"],
#    )


#~~~~~~~ Retriever code

#retriever=vectordb.as_retriever(search_kwargs={"k":10, "fetch_k":50}, search_type="mmr")
retriever=vectordb.as_retriever(search_kwargs={"k":25})

#~~~~~~ if using MultiQueryRetriever
#from langchain.retrievers.multi_query import MultiQueryRetriever
#retriever_multiquery = MultiQueryRetriever.from_llm(
#  retriever=retriever, llm=llm,
#)

#~~~ improve prompt for Admissions
from utility import convert_messages_to_llm_format
from llm import get_completion_by_messages
def improve_prompt(user_prompt):
    sys_prompt = """Your task is to improve a user question that will be fed into LangChain's MultiQueryRetriever. 
    1. Identify key terms in the user question, such as the student's certification, the admission exercise, etc.
    2. Assume that the user question should be about post-secondary school admissions in Singapore, \
        so replace key terms accordingly so that the MultiQueryRetriever will be fed with the right question.
    3. Improve the user question so that MultiQueryRetriever retrieves the documents that are relevant to these key terms.
    4. Make your improved question more concise.
    5. Respond with only the prompt."""
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
  retriever=retriever, llm=llm, 
)

#~~~~~~~~ RetrievalQA code
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever_multiquery,
    return_source_documents=True, # Make inspection of document possible
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, 
                       "verbose":False,
                       "memory":memory,},
    
)

#~~~~~~~~~~~~~~~~ Test codes
#~~~~~~~~ Invoke and Response
#response = qa_chain.invoke("I have a N-level cert, what admission exercises am i eligible for?")
#print(response.get('result'))

#print(memory)
#~~~ use this to clear the buffer memory when starting over
#memory.clear()


#~~~~~~~~ Invoke and Response - 2
#response = qa_chain.invoke("only those that lead me to join a poly.")
#print(response.get('result'))

#~~~~~~~~ Invoke and Response - 3
#response = qa_chain.invoke("yes, please.")
#print(response.get('result'))

#~~~~~~~~ Invoke and Response - 4
#response = qa_chain.invoke("PFP.")
#print(response.get('result'))


#print(len(response.get('source_documents')))

#for i in range(len(response.get('source_documents'))):
#    print(f"""\n\n###{i}:\n{response.get('source_documents')[i].metadata.get('description')}\n{response.get('source_documents')[i].metadata.get('source')}\n""")


#~~~~~~~~ Include today's date in prompt for responses regarding admission matters to be date senstitive. 
from datetime import datetime

@st.cache_data(ttl=300)
def admissions_invoke_question(user_message):
    today_date=datetime.today().strftime('%d/%m/%Y')
    user_message = improve_prompt(user_message)
    print(user_message)
    response = qa_chain.invoke(f"{user_message}. Today's Date: {today_date}.")
    response_text = response.get('result')
    #print(memory.load_memory_variables({}))
    return response_text

def clear_memory():
   memory.clear()

def get_chat_history():
   return memory.buffer

def get_chat_history_msg():
   formatted_messages = convert_messages_to_llm_format(memory.chat_memory.messages)
   return formatted_messages