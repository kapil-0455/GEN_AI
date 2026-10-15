
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

chat_model = ChatOllama(model="llama3.2:1b")


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City of the person")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "Generate the name, age and city of a fictional "
        "{place} person.\n{format_instruction}"
    ),
    input_variables=["place"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain = template | chat_model | parser

result = chain.invoke({"place": "Hyderabad"})

print(result)
print(result.name)
print(result.age)
print(result.city)
