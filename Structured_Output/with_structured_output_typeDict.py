from langchain_ollama import ChatOllama
from typing import TypedDict , Annotated , Literal, Optional

model = ChatOllama(model='llama3.2:1b')


# Annotated mae pta thodi jyada detail bta sktae hain LLM kaa ouptut jyada achae krnae kae liyae 

# literal -> to give options
class Review(TypedDict):
    summary : Annotated[str , "A breif summary of review"]
    sentiment : Annotated[Literal["positive" , "negative"] , "The sentiment of the review"] # yhn pr yae literal use krta 

    pros : Annotated[list[str] , "List of pros must include build quality , design , performance , value for money , packaging , ease of use , durability , appearance , functionality , convenience , materials , price , setup , features , user experience , brand , quality , value , etc"]
    cons : Annotated[list[str] , "List of cons must include build quality , design , performance , value for money , packaging , ease of use , durability , appearance , functionality , convenience , materials , price , setup , features , user experience , brand , quality , value , etc"]

    name : Annotated[Optional[str] , "Give the name of reviewer"]
# yae kya krr rha hae kuch alag prompt genertae krkae uskoa attch krr deta hae jsisae hmae btaan nhi pdta kei hmae kya krna hae LLM ko 

# sentiment or summary apnae aap hei nikal deta hae oskae baad hmarae schema ko match krta hae  
structured_output_model = model.with_structured_output(Review)

response = structured_output_model.invoke('''
⭐⭐⭐⭐⭐ Excellent Product with Great Build Quality and Performance

I am extremely satisfied with this product and my overall experience has been excellent. From the moment I received it, the product impressed me with its quality, design, and finishing. The build quality feels solid and durable, which gives confidence that it will last for a long time with proper use. The design is modern, attractive, and practical, making it suitable for everyday use.

In terms of performance, the product works smoothly and delivers exactly what I expected. It is easy to use, convenient to handle, and performs its intended functions efficiently. The overall user experience is quite good, and I have not faced any major issues while using it. The attention to detail and the quality of the materials make it feel like a well-designed product.

Another thing I appreciate is the value for money. Considering the quality, features, and performance, I believe this product is worth the price. The packaging was satisfactory, and the product arrived in good condition. Setting it up and getting started was also straightforward, so there was no unnecessary hassle.

I also like that the product offers a good balance between functionality, durability, and appearance. Whether you want to use it regularly at home, in the office, or for personal purposes, it can be a useful addition to your setup. Overall, the product meets my expectations and provides a satisfying experience.

I would recommend this product to anyone looking for a reliable, well-built, and reasonably priced option. Based on my overall experience, I am happy with this purchase and would consider buying from the same brand again in the future. Definitely a good product that offers quality and convenience at a reasonable price!''')

# print(response)
# print(response['summary'])
# print(response['sentiment'])
print(response['name'])



### Only one cone hae typrDict kaa validation nhi hae 