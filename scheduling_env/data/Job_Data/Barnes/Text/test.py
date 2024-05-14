import os
path = './'
files = os.listdir(path)
files.pop()
record = [0 for _ in range(100111)]

for p in files:
    with open(p,'r') as f:
        lines = f.readlines()[1::]
        
        lines[0][0:-1].split()

        for line in lines:
            r = line[0:-1].split()
            for i in range(int(r[0])):
                j = 1
                while j<len(r):
                    for k in range(int(r[j])):
                        print(int(r[j+1+k*2]))
                        record[int(r[j+1+k*2])] +=1
                    j = j+ int(r[j])*2+1

print(record)


