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
    
    def append(self,node:any) -> None:
        if self._head is None:
            node.prev = node.next = None
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail.next.prev = self._tail
            self._tail = self._tail.next
    def prepend(self, node:any) -> None:
        if self._head is None:
            self._head = self._tail = node
        else:
            self._head.prev = node
            self._head.prev.next = self.head
            self._head = self.head.prev
    def delete(self,node:Node):
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node is self._head:
            self._head = node.next
        if node is self._tail:
            self._tail = node.prev
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