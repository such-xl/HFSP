with open('mt10c1.fjs','r') as f:
    job_n,maine_n = map(int,f.readline().split()[0:-1])
    print(job_n,maine_n)
    for job_i, line_str in enumerate(f, start=1):
        line = list(map(int, line_str.split()))
        print(f'作业{job_i}')
        i = 1
        r = 1
        while i < len(line):
            s = f'工序{r} '
            for j in range(line[i]):
                s +=f'机器{line[i+1+j*2]} 耗时{line[i+1+j*2+1]} || '
            print(s)
            r += 1
            i += (1+line[i]*2)