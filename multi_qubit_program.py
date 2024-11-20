import gate_operations as go
import numpy as np
import scipy.sparse
from scipy.sparse import lil_array
from scipy.sparse import csr_matrix





# INPUT SPACE
qubit_amount = 4 # all qubits are initilized as |0〉


# state prep |0000...0000〉
state = np.zeros((2**qubit_amount,))
state[0] = 1

# non sparse
H1 = go.single_qubit_gate_to_full_gate(go.H(), qubit_amount, 2)
#print(H1)

x = 1/np.sqrt(2)

H_sparse = lil_array((2, 2),None,complex)   # 1  1
H_sparse[0,0] = x              # 1 -1
H_sparse[0,1] = x
H_sparse[1,0] = x
H_sparse[1,1] = -x

H_sparse = H_sparse.tocsr()

Id_sparse = lil_array((2, 2),None,complex)
Id_sparse[0,0] = 1   
Id_sparse[0,1] = 0
Id_sparse[1,0] = 0
Id_sparse[1,1] = 1

Id_sparse = Id_sparse.tocsr()



full_sparse = scipy.sparse.kron(Id_sparse, H_sparse)


#print(full_sparse.toarray())

#print(full_sparse.toarray() == H1)


# non sparse
Y = go.single_qubit_gate_to_full_gate(go.Y(), qubit_amount, 2)
#print(Y)

Y_sparse = lil_array((2, 2),None,complex)  
Y_sparse[0,0] = 0             
Y_sparse[0,1] = -1j
Y_sparse[1,0] = 1j
Y_sparse[1,1] = 0

Y_sparse = Y_sparse.tocsr()

full_sparse = scipy.sparse.kron(Id_sparse, Y_sparse)


#print(full_sparse.toarray())

#print(full_sparse.toarray() == Y)

sol1 = go.gate_operation(state, Y)
print(sol1)

state_sparse = scipy.sparse.csr_matrix(state)

#print(state_sparse)
#print(full_sparse)

sol2 = go.gate_operation(csr_matrix(state.reshape(-1, 1)), csr_matrix(Y))
print(sol2.toarray())

#print(type(csr_matrix(state.reshape(-1, 1))))


