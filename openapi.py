from langchain.chains.openai_functions.openapi import get_openapi_chain

chain = get_openapi_chain(
    "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/"
)
chain("What are some options for a men's large blue button down shirt")

