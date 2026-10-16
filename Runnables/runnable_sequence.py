from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="llama3.2:1b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="give me joke about this topic {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(template="explain the joke {joke}", input_variables=["joke"])
chain = RunnableSequence(
    prompt1,
    chat_model,
    parser,
    prompt2,
    chat_model,
    parser
)

print(chain.invoke({"topic": "AI"}))