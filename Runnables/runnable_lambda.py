from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="llama3.2:1b")

parser = StrOutputParser()

def word_count(text:str) -> str:
    return len(text.split())


prompt1 = PromptTemplate(
    template="give me joke about this topic {topic}", input_variables=["topic"]
)

joke_gen_chain = RunnableSequence(prompt1, chat_model, parser)

parllel_Chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "total words in joke": RunnableLambda(word_count) ,
    }
)

final_chain = RunnableSequence(joke_gen_chain | parllel_Chain)


res = final_chain.invoke({"topic": "AI"})

# print(res)
final_res = f'''Joke is : {res['joke']}\n Total words in joke is : {res['total words in joke']}'''
print(final_res)