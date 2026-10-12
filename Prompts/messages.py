from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2:1b", temperature=0.7)

# system message ->> system level messgae ekdum shurur wala message eg, you are knowledgeable doctor and must answer in marathi
# human message ->> user wala message 
# ai message ->> ai wala message 

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about langchain"),
]

res = model.invoke(messages)
messages.append(AIMessage(content=res.content))

print(messages)