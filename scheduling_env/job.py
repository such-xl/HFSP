from .utils import Node
class Job(Node):
    def __init__(self,id:int,process_num:int,process_list:list) -> None:
        super().__init__(None)
        self._id = id #job序号,从1开始
        self._process_num = process_num         #job工序数
        self._process_list = process_list       #job工序列表[{机器1:加工时间1,机器2:加工时间2},...{}]
        self._progress = 1                       # 加工进度 代表第progess道工序待加工，0 代表加工完成
        self._status = 0                        # 0 已完成   1 加工中  2待加工
        self._machine_id = 0                     # 正在加工该job的机器id，0表示目前没有被加工
        self._t_process = 0                    # 当前工序需被加工的时间
        self._t_processed = 0                  # 当前工序已经被加工时间

    def show(self):
        print(f'作业{self._id} 工序数{self._process_num}')
        for i,p in enumerate(self._process_list,start=1):
            print(f'工序{i}')
            print(p)
    
    def get_t_process(self, machine_id):
        return self._process_list[self._progress-1][machine_id]
    
    # 判断当前工序是否可被agent_i执行
    def match_machine(self,machine_id) -> bool:
        return machine_id in self._process_list[self._progress-1]
    
    # 将job装载至machine
    def load_to_machine(self,machine_id):
        self._machine_id = machine_id
        self._t_process = self.get_t_process(machine_id)
        self._t_processed = 0
        self._status = 1
    #加工一个时序
    def run_a_time_step(self):
        self._t_processed += 1
        print(f'机器{self._machine_id}加工作业{self._id} 1s',end=' ')
        if self._t_processed == self._t_process:        #当前工序加工完成
            self._t_processed = 0
            self._t_process = 0
            self._machine_id = 0
            self._progress +=1
            print(f'工序完成',end=' ')
            if self._progress == self._process_num+1:    # 最后一道工序加工完成
                self._progress = 0
                self._status = 0
            else:
                self._status = 2
            print()
    
    @property
    def id(self):
        return self._id
    @property
    def process_num(self):
        return self.process_num
    @property
    def process_list(self):
        return self._process_list
    @property
    def progress(self):
        return self._progress
    @property
    def status(self):
        return self._status
    @property
    def machine_id(self):
        return self._machine_id
    @property
    def t_process(self):
        return self._t_process
    @property
    def t_processed(self):
        return self._t_processed
