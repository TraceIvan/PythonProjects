#派生类可以继承父类的公有成员,但不能继承私有成员
#多继承:如果父类中有相同的方法名,而在子类中使用时没有指定父类名,则Python解释器将从左向右按顺序进行搜索
import types
class Person(object):#基类必须继承于object，否则在派生类中无法使用super()函数
    def __init__(self,name='',age=20,sex='man'):
        self.setName(name)
        self.setAge(age)
        self.setSex(sex)

    def setName(self,name):
        if type(name)!=str:
            print('name must be string.')
            return
        self.__name=name

    def setAge(self,age):
        if type(age)!=int:
            print('age must be integer.')
            return
        self.__age=age

    def setSex(self,sex):
        if sex!='man' and sex!='woman':
            print('sex must be "man" or "woman"')
            return
        self.__sex=sex

    def show(self):
        print(self.__name)
        print(self.__age)
        print(self.__sex)

class Teacher(Person):
    def __init__(self,name='',age=30,sex='man',department='Computer'):
        #调用基类构造方法初始化基类私有数据成员
        super(Teacher,self).__init__(name,age,sex)
        #Person.__init__(self,name,age,sex)#也可以这样初始化基类私有数据成员
        self.setDepartment(department)

    def setDepartment(self,department):
        if type(department)!=str:
            print('department must be a string.')
            return
        self.__department=department

    def show(self):
        super(Teacher,self).show()
        print(self.__department)

class Student(Person):
    def __init__(self,name='',age=16,sex='man',major='Computer'):
        super(Student,self).__init__(name,age,sex)
        self.setMajor(major)
    def setMajor(self,major):
        if type(major)!=str:
            print('major must be string.')
            return
        self.__major=major
    def show(self):
        super(Student,self).show()
        print(self.__major)

if __name__=='__main__':
    zhangsan=Person('Zhang San',19,'man')
    zhangsan.show()
    lisi=Teacher('Li si',32,'man','Math')
    lisi.show()
    ww=Student('Wang Wu',14,'man','physic')
    ww.show()
