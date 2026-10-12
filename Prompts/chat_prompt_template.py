from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_ollama import ChatOllama


# But aise yae code kaam nhi krega 
# hmae kuch alag trikae sae pass krna hoga
chat_template= ChatPromptTemplate(
    [
        SystemMessage(content="You are a helpful {domain} expert. "),
        HumanMessage(content='Explain in simple terms ,what is {topic}')
    ]
)
prompt = chat_template.invoke({'domain':'AI', 'topic':'Generative AI'})
print(prompt)