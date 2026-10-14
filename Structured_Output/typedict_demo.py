from typing import TypedDict


class Person(TypedDict):
    name : str
    age : int


# glt bhi daal sktae hain  
new_person: Person = {"name":"Bob", "age": '35'}

print(new_person) 