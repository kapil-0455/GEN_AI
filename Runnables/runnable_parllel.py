from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="llama3.2:1b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Give me tweet on given topic {topic}",
    input_variables = ['topic']
)


prompt2 = PromptTemplate(
    template = "Give me linkedin post on given topic {topic}",
    input_variables = ['topic']
)

chain = RunnableParallel({
    'tweet' : prompt1 | chat_model | parser,
    'linkedin' : prompt2 | chat_model | parser
})

result = chain.invoke({'topic' : 'AI'})
print(result['tweet'])
print(result['linkedin'])
