from functools import reduce
from operator import mul
# a = [1,2,2,2]
a = (1,2,2,2)
base = 32

for i in range(4):
    if i == 0:
        print(base)
    else:
        # print(sum(list(map(lambda x: x, a[:i])) * base))
        print(f"In chann: {reduce(mul, a[:i]) * base}, Out chann: {reduce(mul, a[:i+1]) * base}")