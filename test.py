from scheduling_env.utils import DoublyLinkList,Node

link_list = DoublyLinkList()
dd_list = DoublyLinkList()
for i in range(1,10):
    link_list.append(Node(i)) 

link_list.print_list()

head = link_list.head
while head:
    if head.data % 2 == 0:
        next_node = head.next
        link_list.disengage_node(head)
        dd_list.append(head)
        head = next_node
        continue
    head = head.next
link_list.print_list()
dd_list.print_list()

c,d=1,2
arr = [1,2]
arr2 = arr
def exchange(a,b):
    print(a,b)
    a ^= b
    b ^= a
    a ^= b
    print(a,b)
def append(arr):
    arr.append(3)
    print(arr)
append(arr)
print(arr2)
arr2 = [1,2]
print(arr)
print(arr2)
exchange(c,d)
print(c,d) 