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

#~~~~~~~~ If using HTMLSectionSplitter Code
#~~~~ RecursiveUrlLoader gives metadata['title'] but HTMLSectionSplitter.split_documents expects metadata['Title'] 
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


#~~~~~~~~ slitter: SemanticChunker 
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
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",input_key="question")

#~~~~~~~~ Prompt Template code
from langchain.prompts import PromptTemplate


# Build prompt
template = """
Previous conversation:
{chat_history}

You are an expert in Post-Secondary School Education admission matters in Singapore.
Use the following pieces of context, delimited by <context> to answer the question at the end. \
Note that Direct-Entry-Scheme to Polytechnic Programme (DPP) and Polytechnic Foundation Programme (PFP) \
    are also Admission Exercises.
If the question is not about Post-Secondary School Education admission matters in Singapore, \
    remind the user what your job is and provide an example what he/she can ask.  
If you don't know the answer, just say that you don't know, do not make up answers. \
Do not make up admission exercises that do not exist. 
Be polite. Keep the answer as concise as possible. 
Add a line break. 
Think about what the user might want to ask about next \
    and suggest with 'Would you also like to find out...' after that.

<context>
{context}
</context>

Question:{question}
Helpful Answer:
Would you also like to find out:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
#QA_CHAIN_PROMPT = PromptTemplate(
#    template=template,
#    input_variables=["chat_history", "context", "question"],
#    )


#~~~~~~~ Retriever code

retriever=vectordb.as_retriever(search_kwargs={"k":5, "fetch_k":25}, search_type="mmr")

#~~~~~~ if using MultiQueryRetriever
#from langchain.retrievers.multi_query import MultiQueryRetriever
#retriever_multiquery = MultiQueryRetriever.from_llm(
#  retriever=retriever, llm=llm,
#)


#~~~~~~~~ RetrievalQA code
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True, # Make inspection of document possible
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, 
                       "memory":memory,},
    
)

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

from datetime import datetime

def invoke_question(user_message):
    today_date=datetime.today().strftime('%d/%m/%Y')
    response = qa_chain.invoke(f"{user_message}. Today's Date: {today_date}.")
    print(memory.load_memory_variables({}))
    return response.get('result')