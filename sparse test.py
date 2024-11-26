from scipy.sparse import lil_array
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand
import numpy as np
from scipy.sparse import csr_matrix
import gate_operations as go
import random


chunk_size = 3

qubit_amount = 8

control = random.randint(1, qubit_amount)

base = control - control%chunk_size

if base == control and base != 1:
    base = base - chunk_size

target = control
if base + 1 != qubit_amount:
    while target == control or target > qubit_amount:
        target = base + random.randint(1, chunk_size)


print(base)

print(control)
print(target)

