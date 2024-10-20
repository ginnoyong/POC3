import recursiveurlloader

from langchain_community.document_loaders import RecursiveUrlLoader
import re
from bs4 import BeautifulSoup

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, features="html.parser")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

loader_courses = RecursiveUrlLoader(
    #"https://www.np.edu.sg/schools-courses/academic-schools",
    "https://www.moe.gov.sg/coursefinder/coursedetail",
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

docs_courses = loader_courses.load()

#~~~~~~~~ Embeddings code
from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

#~~~~~~~~ ChatOpenAI llm
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

###############
import os
os.environ["SERPER_API_KEY"] = "2bdc6f17f95bd25365f22b96cea3f14895d480e1"

from langchain.utilities import GoogleSerperAPIWrapper
googleSerperAPIWrapper = GoogleSerperAPIWrapper()

from langchain.retrievers.web_research import WebResearchRetriever

from langchain_chroma import Chroma
vectordb_courses = Chroma(embedding_function=embeddings_model, collection_name='courses2', persist_directory='./vector_db')
web_search_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectordb_courses, llm=llm, search=googleSerperAPIWrapper
)

###############

#~~~~~~~~ Chroma Vector Store code
#from langchain_chroma import Chroma
#vectordb_courses = Chroma.from_documents(docs_courses, embeddings_model, collection_name='courses', persist_directory='./vector_db')
#print(len(docs_courses))
#retriever_courses = vectordb_courses.as_retriever(search_kwargs={"k":5, "fetch_k":25}, search_type="mmr")

#~~~~~~~~ Prompt Template code
from langchain.prompts import PromptTemplate


# Build prompt
template = """
Previous conversation:
{chat_history}

You are an expert in Post-Secondary School Education schools and coures in Singapore.
Use the following pieces of context, delimited by <context> to answer the question at the end. 

Note that: 
Aggregate refers to aggregate scores used in Joint Admission Exercise (JAE).
There are several types of aggregate scores: ELR2B2-A, ELR2B2-B, ELR2B2-C, ELR2B2-D, L1R5, etc.
A student's aggregate score is lower when his/her Secondary School grades are better. \
Conversely, a student's aggregate score is higher when his/her Secondary School grades are worse.
The aggregate score range of a course provides a reference of the aggregate scores \
    that students will need to acquire to be accepted into the course. For example: \
    Aggregate score range of Accountancy course in Nanyang Polytechnic is 6 - 10 (6 to 10). \
    If student A has an aggregate score of 5, student B as 8, student C has 11, \
    and they apply for the Accountancy course in Nanyang Polytechnic, \
    Student A will have high chance of being accepted because his/her aggregate score is lower than the course's Aggregate Score range. \
    Student B will have medium chance of being accepted because his/her aggregate score is within the course's Aggregate Score range. \
    Student C will have low high chance of being accepted because his/her aggregate score is within the course's Aggregate Score range.

If the question is not about Post-Secondary School Education schools and courses in Singapore, \
    remind the user what your job is and provide an example what he/she can ask.  
If you don't know the answer, just say that you don't know, do not make up answers. \
Do not make up courses that do not exist. 
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

#~~~~~~~~ ConversationBufferMemory code
from langchain.memory import ConversationSummaryMemory
# memory = ConversationBufferMemory(memory_key="chat_history",input_key="question")
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",input_key="question")

#~~~~~~~~ RetrievalQA code
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=web_search_retriever,
    return_source_documents=True, # Make inspection of document possible
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, 
                       "memory":memory,},
)

#~~~~~~~~ Invoke and Response
response = qa_chain.invoke("List all accountancy courses.")
print(response.get('result'))

print(memory)
