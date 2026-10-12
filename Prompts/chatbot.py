from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ==========================================
# OLLAMA LOCAL MODEL (FREE - No API key needed)
# ==========================================

chat_model = ChatOllama(
    model="llama3.2:1b",
    temperature=0.7,
)

chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]



while True:
    user_input = input('You : ')
    if user_input == 'exit':
        break
    chat_history.append(HumanMessage(content=user_input))
    res = chat_model.invoke(chat_history)
    chat_history.append(AIMessage(content=res.content))
    print('AI : ', res.content)

print(chat_history)