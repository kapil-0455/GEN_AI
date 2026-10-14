from pydantic import BaseModel, EmailStr, Field
from typing import Optional

# khud sae smjhta hae apnae kya data bheja hae 
# eg -> '32' ko 32 krr dega implicit type conversion

class Student(BaseModel):
    name : str = 'Kapil'
    age : Optional[int] = None # set optinal values 
    email : EmailStr
    cgpa : float = Field(gt = 0 , lt=10 , default = 5.5 , description = "cgpa of student must be between 0 to 10")


new_student = {'name' : 'Kapil' , 'age' : 22 , 'email' : 'kapil@gmail.com'}

student = Student(**new_student)

print(student.age)
print(student.name)
print(student.email)

# now we can make dict json or list of objects 

student_json = student.model_dump_json()
print(student_json)