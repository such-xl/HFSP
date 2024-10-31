dic = {1:3,2:4,5:9}
lenth = 10
flag = True
m_id = 2
arr = [dic.get(i, 0) if not flag else dic.get(m_id) if i==m_id-1 else 0 for i in range(lenth)]
a1 = [1,2,3]
a2 = [4,5]
a3 = [6,7,8,9,0]
a1 += a2+a3
print(a1)
