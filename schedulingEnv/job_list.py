'''
    作业链表类
'''
from utils import DoublyLinkList
from job import Job
class JobList(DoublyLinkList):
    def __init__(self) -> None:
        super().__init__()
        self._job_list: list[Job] = []
        self._job_num = 0
        self._machine_num = 0
        self._head = None
    
    # 从链表头部插入job
    def prepend(self, job: Job):
        job.next = self._head
        self._head = job
        self._job_num += 1
    # 从文件中获取作业信息
    def decode_job_flie(self,path:str):
        with open(path,'r') as f:

            _,self._machine_num = map(int,f.readline().split()[0:-1])

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
 
                self.prepend(Job(job_id,r-1,procs))
    @property
    def job_num(self):
        return self._job_num    
    @property
    def machine_num(self):
        return self._machine_num
    @property
    def head(self):
        return self._head    

jl = JobList()
jl.decode_job_flie('data/Job_Data/Barnes/Text/mt10c1.fjs')
c_n = jl._head
while c_n:
    c_n.show()
    c_n = c_n.next
print(jl._job_num)
