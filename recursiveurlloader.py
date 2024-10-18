#~~~~~~~~ Loader Code
from langchain_community.document_loaders import RecursiveUrlLoader
import re
from bs4 import BeautifulSoup

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, features="html.parser")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

loader = RecursiveUrlLoader(
    "https://www.moe.gov.sg/post-secondary/admissions",
    #"https://docs.python.org/3.9/",
    # extractor=bs4_extractor,
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
#print(docs[0].metadata)

#~~~~~~~~ If using HTMLSectionSplitter Code
#~~~~~ RecursiveUrlLoader gives metadata['title'] but HTMLSectionSplitter.split_documents expects metadata['Title'] 
for doc in docs:
    doc.metadata['Title']=doc.metadata.get('title')
# print(docs[0].metadata)

from langchain_text_splitters import HTMLSectionSplitter
headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]
html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
html_splitted_docs = html_splitter.split_documents(documents=(docs))
#print(len(html_splitted_docs))

#~~~~~~~~ Embeddings code
from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

#~~~~~~~~ If using SemanticChunker Code
#from langchain_experimental.text_splitter import SemanticChunker

# Create the text splitter
#semantic_text_splitter = SemanticChunker(embeddings_model)

# Split the documents into smaller chunks
#semantic_splitted_documents = semantic_text_splitter.split_documents(docs)
#print(len(semantic_splitted_documents))

#~~~~~~~~ Chroma Vector Store code
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

vectordb = Chroma.from_documents(html_splitted_docs, embeddings_model, collection_name='admissions', persist_directory='./vector_db')
# vectordb.reset_collection()
# print(f"########:{len(vectordb.get()['documents'])}")

#~~~~~~~~ Prompt Template code
from langchain.prompts import PromptTemplate

# Build prompt
template = """
Use the following pieces of context, delimited by <context> to answer the question at the end. \
If you don't know the answer, just say that you don't know, do not make up answers. \
Keep the answer as concise as possible. 
Always say "Hope this answers your question!" at the end of your answer. 
<context>
{context}
</context>
Question:{question}
Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

#~~~~~~~~ ChatOpenAI and MultiQueryRetriever code
from langchain_openai import ChatOpenAI
#from langchain.retrievers.multi_query import MultiQueryRetriever

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
retriever=vectordb.as_retriever(search_kwargs={"k": 20}, search_type="mmr")

#retriever_multiquery = MultiQueryRetriever.from_llm(
#  retriever=retriever, llm=llm,
#)


#~~~~~~~~ RetrievalQA code
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True, # Make inspection of document possible
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

#~~~~~~~~ Invoke and Response
response = qa_chain.invoke("is the PFP programme admission exercise for N or o-level students?")

#print(response)
#print(len(response.get('source_documents')))

print(response.get('result'))

#for i in range(len(response.get('source_documents'))):
#    print(f"""\n\n###{i}:\n{response.get('source_documents')[i].metadata.get('description')}\n{response.get('source_documents')[i].metadata.get('source')}\n""")