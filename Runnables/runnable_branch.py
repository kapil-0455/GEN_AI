from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
    RunnableBranch,
)
from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="llama3.2:1b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Give me the deatiled report on this topic {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Give me a short summary of given topic {topic}", input_variables=["topic"]
)

report_gen = RunnableSequence(prompt1 | chat_model | parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300,RunnableSequence(prompt2 , chat_model , parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen | branch_chain)

print(final_chain.invoke({"topic": "AI"}))
