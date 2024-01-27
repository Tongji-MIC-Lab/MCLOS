import numpy as np
from scipy import stats
f = open("/remote-home/share/zhangyuxuan/FSS-master/utils/results.txt",encoding = "utf-8")
#输出读取到的数据
lines = f.readlines()
n = len(lines)
# print(n)
base = []
novel = []
for i in range(n):
    base.append(float(lines[i].split(',')[5]))
    novel.append(float(lines[i].split(',')[6]))
b_a, n_a = np.mean(base),np.mean(novel)
print(b_a,n_a)
print(stats.hmean([b_a,n_a]))
f.close()