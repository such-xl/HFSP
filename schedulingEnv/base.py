import asyncio
import random
import time
class Machine:
    def __init__(self,ID:int,TYPE:str) -> None:
        self.ID = ID
        self.TYPE = TYPE
        self.statu = 1
        self.currentTool = -1
    def getinfo()->dict:
        pass
    def __swichTool(t=1)->bool:
        asyncio.sleep(t)
        print('切换工具')
        return True
    def process()->bool:
        asyncio.sleep(2)
        print('处理成功')
    async def run(self)->bool:
        while True:
            flag = random.randint(0,1)
            ct = time.time()
            print('当前时间',ct-bt)
            if flag:
                await asyncio.sleep(1)
                print(self.ID,'切换的工具')
            else:
                await asyncio.sleep(2)
                print(self.ID,'处理成功')
    def print(self) -> None:
        print(f'ID:{self.ID},TYPE:{self.TYPE}')

class Process:
    def __init__(self,name,processtime) -> None:
        self.name = name
        self.processtime = processtime
class Product:
    def __init__(self,ID:int,name:str,process:any) -> None:
        self.ID = ID
        self.name = name
        self.process=process  
class Order:
    def __init__(self,id,startTime:any,deadline:any,profit:float,deferredPenalty:float,product:Product,num:int)->None:
        self.id = id
        self.startTime = startTime
        self.deadLine = deadline
        self.profit = profit
        self.deferredPanalty = deferredPenalty
        self.product = Product
        self.num = num
 
"""
    床脑（开料，铣型，贴皮）
    床头（开料，铣型，抛光，贴皮)
    床尾屏（开料，铣型，抛光）
    process:{
        0:开料, 1:铣型, 2:抛光 3:贴皮
    }
"""
MachineTYPE = {0:'开料', 1:'铣型', 2:'抛光', 3:'贴皮'}
ProductTYPE = {0:'床脑',1:'床头',2:'床尾屏'}


p0 = Product(0,ProductTYPE[0],[Process(MachineTYPE[0],3),Process(MachineTYPE[1],4),Process(MachineTYPE[3],4)])
p1 = Product(1,ProductTYPE[1],[Process(MachineTYPE[0],3),Process(MachineTYPE[1],4),Process(MachineTYPE[2],2),Process(MachineTYPE[3],4)])
p2 = Product(2,ProductTYPE[2],[Process(MachineTYPE[0],3),Process(MachineTYPE[1],4),Process(MachineTYPE[2],3)])

m0_0 = Machine(0,MachineTYPE[0])
m0_1 = Machine(1,MachineTYPE[0])
m1_2 = Machine(2,MachineTYPE[1])
m1_3 = Machine(3,MachineTYPE[1])
m2_4 = Machine(4,MachineTYPE[2])
m3_5 = Machine(5,MachineTYPE[3])

o_0 = Order(0,0,300,1000.0,-1,p0,10)
o_1 = Order(1,0,200,500.0,-2,p1,50)
o_2 = Order(2,0,150,600.0,-1,p2,50)
bt = time.time()
async def main():
    task = [asyncio.create_task(m0_0.run()),asyncio.create_task(m1_2.run())]

    await asyncio.sleep(20)
    for t in task:
        t.cancel()
    await asyncio.gather(*task,return_exceptions=True)
    print('所有机器已经停止')

asyncio.run(main())