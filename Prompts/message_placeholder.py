from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder


# use to retrive and store chatHistory from the file

chat_template = ChatPromptTemplate([
    ('system' , 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name= 'chat_history'),
    ('human' , '{query}')
])

chat_history = []
# load chat history

with open ('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': 'When will my refund be processed?'
})

print(prompt)