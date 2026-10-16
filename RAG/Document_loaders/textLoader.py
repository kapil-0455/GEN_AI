from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="llama3.2:1b")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Summarize the given document:\n{document}", input_variables=["document"]
)

loader = TextLoader("cricket.txt", encoding="utf-8")

docs = loader.load()

chain = prompt | chat_model | parser

print(chain.invoke({'document' : docs[0].page_content}))