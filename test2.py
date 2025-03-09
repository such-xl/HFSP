class A:
    def __init__(self):
        self.arr = [1, 2]

    def pust(self, x):
        self.arr.append(x)


agent = A()
agents = [agent] * 10

for a in agents:
    a.pust(3)
    print(a.arr)
