import numpy as np
import itertools

a=list(itertools.product([0, 1], repeat=16))
b=np.array(a).tolist()

print(tuple(b))