
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat_model = ChatOllama(model="llama3.2:1b")


template = PromptTemplate(
    input_variables = ['topic'],
    template = """
    Generate 5 intresting facts about {topic}
    """
)

parser = StrOutputParser()

chain = template | chat_model | parser

result = chain.invoke({'topic':'AI'})
print(result)

# visualize krnae kae liyae
chain.get_graph().print_ascii()