class Machine:
    def __init__(self,ID:int,TYPE:str) -> None:
        self.ID = ID
        self.TYPE = TYPE
        self.isBroken = False
    def getinfo()->dict:
        pass
    def __swichTool(*agrc)->bool:
        pass
    def process()->bool:
        pass
    def print(self) -> None:
        print(f'ID:{self.ID},TYPE:{self.TYPE}')
class Process:
    def __init__(self,name) -> None:
        self.name = name
class Product:
    def __init__(self,ID:int,name:str,process:any) -> None:
        self.ID = ID
        self.name = name
        self.process=process
    
class Order:
    def __init__(self,startTime:any,deadline:any,profit:float,deferredPenalty:float,detail:any)->None:
        self.startTime = startTime
        self.deadLine = deadline
        self.profit = profit
        self.deferredPanalty = deferredPenalty
        self.detail = detail
 
