'''
    对于job和agent,目前未实现新增
'''
from .utils import DoublyLinkList
from .machine import Machine

class MachineList(DoublyLinkList):
    def __init__(self,machine_num) -> None:
        super().__init__()
        self._machine_num = machine_num
        for id in range(1, self.machine_num+1):
            self.append(Machine(id,[],1,{}))

    @property
    def machine_num(self):
        return self._machine_num
    @machine_num.setter
    def machine_num(self, value):
        self._machine_num = value
'''
ml = MachineList(10)
m = ml.head
while m:
    print(m.id)
    m = m.next
'''