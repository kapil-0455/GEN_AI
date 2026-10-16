from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="llama3.2:1b")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Answer the given question {question} from the document\n{document}", 
    input_variables=[ "question" , "document"]
)

url = "https://www.flipkart.com/samsung-galaxy-book4-metal-intel-core-i3-13th-gen-1315u-8-gb-512-gb-ssd-windows-11-home-np750xgj-lg9in-thin-light-laptop/p/itmc29d559898e6f?pid=COMH4Y6KKRC9YSWE&lid=LSTCOMH4Y6KKRC9YSWERDUMZD&marketplace=FLIPKART&store=6bo%2Fb5g&srno=b_1_1&otracker=browse&fm=organic&iid=6b7a475c-16ce-4c1e-83cb-5ed0ad32cc2a.COMH4Y6KKRC9YSWE.SEARCH&ppt=hp&ppn=homepage&ssid=ungwhptr6o0000001792167235409&ov_redirect=true"

loader = WebBaseLoader(url)
docs = loader.load()


chain = prompt | chat_model | parser

print(chain.invoke({"document": docs , "question" : "What is the price of the laptop?"}))
