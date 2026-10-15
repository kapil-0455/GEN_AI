
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="llama3.2:1b")

schema = [
    ResponseSchema(name="fact_1", description="First fact about the topic"),
    ResponseSchema(name="fact_2", description="Second fact about the topic"),
    ResponseSchema(name="fact_3", description="Third fact about the topic"),
    ResponseSchema(name="fact_4", description="Fourth fact about the topic"),
    ResponseSchema(name="fact_5", description="Fifth fact about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

prompt = PromptTemplate(
    template="""
    Give me the top 5 facts about {topic}.
    {format_instruction}
    """,
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)


chain = prompt | chat_model | parser

result = chain.invoke({"topic": "AI"})

print(result)
