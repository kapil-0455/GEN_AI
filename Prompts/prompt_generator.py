from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=[
        "role",
        "topic",
        "difficulty"
    ],

    template="""
You are a {role}.

Explain the following topic:

Topic: {topic}

Difficulty Level: {difficulty}

Rules:
- Explain clearly.
- Use simple language.
- Give examples.
- If appropriate, provide code.
""",

validate_template = True
)

template.save('template.json')