#1、算术运算符
# x+y：算数加法，列表、元组、字符串合并
# x-y：算数减法
# x*y：乘法，序列重复
# x/y：除法
# x//y：求整商
# -x：负数
# x%y：求余
# x**y：x的y次幂

#2、关系运算符
# x>=y: 大于等于
# x>y: 大于
# x<=y: 小于等于
# x<y: 小于
# x==y: 等于
# x!=y: 不等于

#3、测试运算符
# in、not in: 成员测试运算符
# is、is not: 对象实体同一性测试（地址）

#4、逻辑运算符
# x and y: 逻辑与(只有x为真才会计算y)
# x or y: 逻辑或（只有x为假才计算y）
# not y: 逻辑非

#5、位运算符
# |,^,&,<<,>>,~

#6、集合运算符
# -，&，|：差集，交集，并集

#——————————————————————————————————————
#1、关系运算符可以连续使用
print(1<2<3)
print(1<2>3)
print(1<3>2)

#2、逻辑运算符and 和 or 的短路求值或惰性求值
# x and y: 逻辑与(只有x为真才会计算y)
# x or y: 逻辑或（只有x为假才计算y）

def Join(chList,sep=None):
    return (sep or ',').join(chList)
def Join2(chList,sep=','):
    return sep.join(chList)
#上面该函数用来使用用户指定的分隔符将多个字符串连接成一个字符串，如果没有指定则使用逗号
chTest=['1','2','3','4']
print(Join(chTest))
print(Join(chTest,':'))
print(Join(chTest,' '))



