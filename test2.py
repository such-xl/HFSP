import math
def sigmoid(x,T):
    return 1 / (1 + math.exp(-x/T))
def exp_decay(x,T):
    return math.exp(-x/T)
for i in range(500,2000):
    print(math.log(i))
