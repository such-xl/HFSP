import random
import numpy as np
random.seed(42)
np.random.seed(42)
def Instance_Generator(M_num, E_ave, New_insert):#实时生成算法 作业插入、作业取消、作业操作修改和机器增加等不确定事件
        '''
        :param M_num: Machine Number
        :param Initial_job: initial job number
        :param E_ave
        :return: Processing time,A:New Job arrive time,
                                    D:Deliver time,
                                    M_num: Machine Number,
                                    Op_num: Operation Number,
                                    J_num:Job NUMBER
                                    EL:ergency level of each job
        '''
        job_info_list = []
        Initial_Job_num = 5
        Op_num = [random.randint(1, 20) for _ in range(New_insert + Initial_Job_num)]  # Operation Number of each job
        A1 = [0 for i in range(Initial_Job_num)]
        A = np.random.exponential(E_ave, size=New_insert)
        A = [int(A[i]) for i in range(len(A))]  # New Insert Job arrive time
        A1.extend(A)
        for i in range(Initial_Job_num + New_insert): # i:job id
            Job_i = []
            for j in range(Op_num[i]): # j:operation id OP_num[i],process_num 固定住操作数 15个
                k = random.randint(1, 4)
                T = list(range(M_num))
                random.shuffle(T)
                T = T[0:k + 1]
                proc: dict[int, int] = {}  # 单个工序
                process_time = random.randint(1, 20)
                for M_i in range(M_num):
                    if M_i in T:
                        proc[M_i] = process_time
                Job_i.append(proc)
            
            # Processing_time.append(Job_i)
            job_info_list.append({"id":i, "process_num": Op_num[i], "process_list": Job_i,"Insert_time": A1[i]})
            # job_info.append({"job_id": i, "process_num": r - 1, "process_list": procs})
        return job_info_list,A1
