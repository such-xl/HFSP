import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class Node:
    def __init__(self, data) -> None:
        self._data = data
        self._prev = None
        self._next = None
    @property
    def data(self):
        return self._data 
    @data.setter
    def data(self,data):
        self._data = data
    @property
    def prev(self):
        return self._prev
    @prev.setter
    def prev(self,prev):
        self._prev = prev
    @property
    def next(self):
        return self._next
    @next.setter
    def next(self,next):
        self._next = next
class DoublyLinkList:
    def __init__(self) -> None:
        self._head = None
        self._tail = None
        self._length = 0 
    def append(self,node:any) -> None:
        if self._head is None:
            node.prev = node.next = None
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail.next.prev = self._tail
            self._tail = self._tail.next
        self._length+=1
    def prepend(self, node:any) -> None:
        if self._head is None:
            self._head = self._tail = node
        else:
            self._head.prev = node
            self._head.prev.next = self.head
            self._head = self.head.prev
        self._length+=1
    def delete(self,node:Node):
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node is self._head:
            self._head = node.next
        if node is self._tail:
            self._tail = node.prev
        self._length -= 1
        del node
    # 将node节点在链上脱离，不删除该节点
    def disengage_node(self,node:Node):
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node is self._head:
            self._head = node.next
        if node is self._tail:
            self._tail = node.prev
        node.prev = node.next = None
        self._length -= 1
        return True
    def print_list(self) -> None:
        temp = self._head
        while temp:
            print(temp.data, end=" ")
            temp = temp.next
        print()
    @property
    def head(self):
        return self._head
    @property
    def tail(self):
        return self._tail
    @property
    def length(self):
        return self._length     

class Plotter:
    def __init__(self, is_live: bool) -> None:
        self.is_live = is_live
        if self.is_live:
            plt.ion()   #开启交互模式
        self.colorset =[
            '#FF69B4', '#FFD700', '#FF4500', '#00FF7F' , '#7FFF00', '#FF1493', '#00BFFF', '#FF8C00',
            '#FFB6C1', '#008080', '#800080', '#9932CC',  '#FF6347', '#BA55D3', '#3CB371', '#a1d99b',
            '#FF00FF', '#a63603', '#228B22', '#6A5ACD',  '#F0E68C', '#4682B4', '#E6E6FA', '#d62728'
            ]
        '''
        self.colorset = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
        '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
        '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', ,
        '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#5254a3',
        '#6b4c9a', '#8ca252', '#bd9e39', '#ad494a', '#636363',
        '#8c6d8c', '#9c9ede', '#cedb9c', '#e7ba52', '#e7cb94',
        '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173',
        '#a55194', '#ce6dbd', '#de9ed6', '#f1b6da', '#fde0ef',
        '#636363', '#969696', '#bdbdbd', '#d9d9d9', '#f0f0f0',
        '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d',
        '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476',
        '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc',
        '#dadaeb', '#636363', '#969696', '#bdbdbd', '#d9d9d9',
        '#f0f0f0', '#a63603', '#e6550d', '#fd8d3c', '#fdae6b',
        '#fdd0a2', '#31a354', '#74c476', '#a1d99b', '#c7e9c0',
        '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb', '#636363',
        '#969696', '#bdbdbd', '#d9d9d9', '#f0f0f0', '#6a3d9a',
        '#8e7cc3', '#b5a0d8', '#ce6dbd', '#de9ed6', '#f1b6da',
        '#fde0ef', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef'
    ]
    '''
    def gant_chat(self,data):
        print(data)
        self.fig, self.ax = plt.subplots()
        machine_num = 0
        for i,info in enumerate(data,start=1):
            for j in info:
                self.ax.barh(f'job{i}',j[2]-j[1],left=j[1],color=self.colorset[j[0]-1])
                machine_num = max(machine_num, j[0])
        patch = []
        for i in range(machine_num):
            patch.append(mpatches.Patch(color=self.colorset[i], label=f'machine{i+1}'))
        self.ax.set_xlabel('time_step')
        self.ax.set_ylabel('job')
        self.ax.set_title('gant chat')
        self.ax.grid(True)
        plt.legend(handles=patch,loc='best')
        plt.tight_layout()
        plt.show()
    def close(self) -> None:
        if self.is_live:
            plt.ioff()
        plt.close()

class StateNorm:
    def __init__(self,job_dim,job_seq_len,machine_dim,machine_seq_len) -> None:
        self.job_dim = job_dim
        self.job_seq_len = job_seq_len
        self.machine_dim = machine_dim
        self.machine_seq_len = machine_seq_len
    def job_seq_norm(self,job_state,ty):
        # job_state:[batch,seq_len,encoding_dim]
        lens = self.job_seq_len
        if ty == 0: #machine的候选作业state
            lens -= 1
        dim = self.job_dim
        mask = np.ones((len(job_state),lens))

        padded_data = np.zeros((len(job_state),lens,dim))
        for i,seq in enumerate(job_state):
            seq_len = len(seq)
            if seq_len > 0:
                padded_data[i,:seq_len,:] = seq
                mask[i,:seq_len] = 0
        if ty == 0:
            zeros_column = np.zeros((len(job_state), 1))
            # 使用 np.hstack 来水平堆叠原数组和这个全1的列向量
            mask = np.hstack([mask, zeros_column])
        # 当on_job为空时候，mask也是全True,然后通过attention layer后出现nan,所以要将mask全改为False
        all_true_mask = np.all(mask, axis=1)
        mask[all_true_mask, :] = False
        return padded_data,mask


        
