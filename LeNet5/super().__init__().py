class Person(object):
    def __init__(self, name, gender, age):
        self.name = name
        self.gender = gender
        self.age = age


class Student(Person):
    def __init__(self, name, gender, age):
        super(Student, self).__init__(name, gender, age)


# 如果初始化的逻辑与父类的不同，不使用父类的方法，自己重新初始化也是可以的
# class Student(Person):
#     def __init__(self, name, gender, age, school, score):
#         # super(Student,self).__init__(name,gender,age)
#         self.name = name.upper()
#         self.gender = gender.upper()
#         self.school = school
#         self.score = score


s = Student('Alice', 'female', 18)
print(s.age)
print(s.name)
