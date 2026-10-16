from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="llama3.2:1b")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Summarize the given document:\n{document}", input_variables=["document"]
)

loader = PyPDFLoader("dl-curriculum.pdf")

docs = loader.load()

full_text = "\n".join([doc.page_content for doc in docs])
chain = prompt | chat_model | parser
print(chain.invoke({"document": full_text}))
