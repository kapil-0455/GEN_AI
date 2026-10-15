from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

chat_model = ChatOllama(
    model="llama3.2:1b",
    temperature=0.7,
)

# 1s prmpt ->detailed report
template = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=["topic"],
)


# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text . \n {text}",
    input_variables=["text"],
)

# Parser kuch nhi hae bss joo data extract krr rhae thae wooh yae krr dega isska 
# simple language me yehihae 
parser = StrOutputParser()

chain = template | chat_model | parser | template2 | chat_model | parser

result = chain.invoke({"topic": "AI"})

print(result)