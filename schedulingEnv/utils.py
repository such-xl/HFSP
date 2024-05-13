class Node:
    def __init__(self, data) -> None:
        self.data = data
        self.prev = None
        self.next = None
        
class DoublyLinkList:
    def __init__(self) -> None:
        self.head = None
        self.tail = None
    
    def append(self,node:any) -> None:
        if self.head is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail.next.prev = self.tail
            self.tail = self.tail.next
    def prepend(self, node:any) -> None:
        if self.head is None:
            self.head = self.tail = node
        else:
            self.head.prev = node
            self.head.prev.next = self.head
            self.head = self.head.prev
    def delete(self,node:Node):
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node is self.head:
            self.head = node.next
        if node is self.tail:
            self.tail = node.prev
        del node
    def print_list(self) -> None:
        temp = self.head
        while temp:
            print(temp.data, end=" ")
            temp = temp.next
        print()
        