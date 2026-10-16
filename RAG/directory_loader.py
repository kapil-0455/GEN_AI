from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader("books", glob="*.pdf", loader_cls=PyPDFLoader)

# docs = loader.load()
docs = loader.lazy_load()# ekk ekk krkae load krega kaam khtm phir aglae pae jayega 

for document in docs:
    print(document.metadata) 

