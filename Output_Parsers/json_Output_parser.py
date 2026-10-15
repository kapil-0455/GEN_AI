from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.2:1b"
)

parser = JsonOutputParser()
# partial variable -> runtime kae pehlae hei fill hoo jayega 
template = PromptTemplate(
    template="""
    Give me name age or city of fictional person \n {format_instruction}
""",
    input_variables = [],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)
chain = template | model | parser
result = chain.invoke({})
print(result)