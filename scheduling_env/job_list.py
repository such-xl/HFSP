'''
    作业链表类
'''
from .utils import DoublyLinkList
from .job import Job
class JobList(DoublyLinkList):
    def __init__(self) -> None:
        super().__init__() 
    # 从文件中获取作业信息
    def decode_job_flie(self,path:str):
        machine_num: int = 0
        with open(path,'r') as f:

            _,machine_num = map(int,f.readline().split()[0:-1])

            for job_id, line_str in enumerate(f, start=1):   
                        
                line = list(map(int, line_str.split()))

                i,r = 1,1

                procs: list[dict[int,int]] = []         # 工序列表
                while i < len(line):
                    #s = f'工序{r} '                                                                                                                                                                 
                    proc: dict[int,int] = {}            # 单个工序
                    for j in range(line[i]):
                        #s +=f'机器{line[i+1+j*2]} 耗时{line[i+1+j*2+1]} || '
                        proc[line[i+1+j*2]]=line[i+1+j*2+1]

                    procs.append(proc)
                    r += 1
                    i  += (1+line[i]*2)
                self.append(Job(id=job_id,process_num=r-1,process_list=procs,insert_time=0))
        return machine_num
                

