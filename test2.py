def my_function(x1, x2, x3, x4, x5,x6):
    pass  # 这里可以添加函数的实际功能
    print(x1,x2,x3,x4,x5,x6)
s1 = ('a1', 'b1', 'c1')
s2 = ('a2', 'b2', 'c2')

# 分别从两个元组中取出所有需要的元素
params = s1 + s2[:-1]

# 调用函数，并通过星号操作符将元组解构为独立的参数
my_function(*(s1+s2[:-1]),'k')