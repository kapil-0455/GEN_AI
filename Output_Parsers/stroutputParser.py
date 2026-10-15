from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

chat_model = ChatOllama(
    model="llama3.2:1b",
    temperature=0.7,
)

# 1s prmpt ->detailed report 
template = PromptTemplate(
   template = "Write a detailed report on the following topic: {topic}",
   input_variables = ['topic']
)


# 2nd prompt -> summary
template2 = PromptTemplate(
   template = "Write a 5 line summary on the following text . \n {text}",
   input_variables = ['text']
)

prompt1 = template.invoke(
    {'topic':'AI'}
)

response = chat_model.invoke(prompt1)

prompt2 = template2.invoke(
    {'text':response.content}
)

result = chat_model.invoke(prompt2)

print(result.content)

