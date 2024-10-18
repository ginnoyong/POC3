from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import pprint

embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

#file_path = "https://www.moe.gov.sg/-/media/files/post-secondary/2025-jae/2025-jae-courses.pdf"
file_path = "https://www.moe.gov.sg/-/media/files/post-secondary/a-guide-to-post-secondary-admissions-exercises.pdf"
loader = PyPDFLoader(file_path)
data = loader.load()

list_of_documents_loaded=[]

list_of_documents_loaded.extend(data)

print("Total documents loaded:", len(list_of_documents_loaded))

# print(list_of_documents_loaded)

# for doc in list_of_documents_loaded:
#    print(doc.page_content[10:110])

# text_splitter = RecursiveCharacterTextSplitter()
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    # chunk_size=500,
    # chunk_overlap=50,
    # length_function=count_tokens
)

# text_splitter = SemanticChunker(embeddings_model)

splitted_documents = text_splitter.split_documents(list_of_documents_loaded)

# print("Total splitted chunks:", len(splitted_documents))
print("Total splitted chunks:", len(list_of_documents_loaded))

for i in range(1,4):
    print(f"""#####\n{splitted_documents[i].page_content}\n\n""")

#for i, chunk in enumerate(splitted_documents):
#   print(f"""{i}: {chunk.page_content[:200]}""")

vectordb = Chroma.from_documents(splitted_documents, embeddings_model, collection_name='courses_points', persist_directory='./vector_db')

# results = vectordb.similarity_search_with_relevance_scores('NYP')

from langchain.prompts import PromptTemplate

# Build prompt
template = """
Use the following pieces of context, delimited by <context> to answer the question at the end. \
If you don't know the answer, just say that you don't know, do not make up an answer. \
Keep the answer as concise as possible. \
Always say "Hope this answers your question!" at the end of your answer.
<context>
N-level cert = Normal course
Two types of Normal course: Normal (Academic) or Normal (Technical)
Normal (Academic) = N(A)
Normal(Technical) = N(T)

{context}
</context>
Question: {question}
Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    ChatOpenAI(model='gpt-4o-mini'),
    retriever=vectordb.as_retriever(k=5),
    return_source_documents=True, # Make inspection of document possible
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

response = qa_chain.invoke("i have n-level cert but i want to join a polytechnic. Which admission exercise should i register for?")
print(response)