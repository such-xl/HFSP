with open('mt10c1.fjs','r') as f:
    lines = f.readlines()
    job_n,maine_n = map(int,lines[0].split()[0:-1])
    print(job_n,maine_n)
    for item in lines[1::]:
        line = list(map(int,item.split()))
        i = 1
        print(line)
        while i < len(line):
            print(f'工件{item.index}')
            for j in range(line[i]):
                print(f'工序{i+1} 机器{line[i+1+j*2]} 耗时{line[i+1+j*2+1]}')
            i += (1+line[i]*2)