from langchain_ollama import ChatOllama
from typing import TypedDict

model = ChatOllama(model='llama3.2:1b')

class Review(TypedDict):
    summary : str
    sentiment : str

# yae kya krr rha hae kuch alag prompt genertae krkae uskoa attch krr deta hae jsisae hmae btaan nhi pdta kei hmae kya krna hae LLM ko 
# sentiment or summary apnae aap hei nikal deta hae oskae baad hmarae schema ko match krta hae  
structured_output_model = model.with_structured_output(Review)

response = structured_output_model.invoke('''
I am really satisfied with this product. The build quality is excellent, and the overall design is both attractive and practical. The product works smoothly and delivers reliable performance. The materials feel durable, and the finishing is impressive for the price. It is easy to use, offers good value for money, and meets my expectations. Packaging was also good, and the product arrived in proper condition. Overall, it is a great purchase, and I would definitely recommend it to anyone looking for a quality product at a reasonable price.''')

print(response)
print(response['summary'])
print(response['sentiment'])