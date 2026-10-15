from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat_model = ChatOllama(model="llama3.2:1b")

template = PromptTemplate(
    input_variables=["topic"],
    template="""
    Generate a detailed report about {topic}
    """,
)

template2 = PromptTemplate(
    input_variables=["text"],
    template="""
    Generate a summary of the following text in 5 points. 
    {text}
    """,
)

parser = StrOutputParser()

chain = template | chat_model | parser | template2 | chat_model | parser

result = chain.invoke({'topic':'AI'})

print(result)
