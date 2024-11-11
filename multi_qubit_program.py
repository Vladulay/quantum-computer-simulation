# Hi
import gate_operations as go
import numpy as np





# INPUT SPACE
qubit_amount = 2 # all qubits are initilized as |0〉









# state prep |0000...0000〉
state = np.zeros((2**qubit_amount,))
state[0] = 1




# GATE DEFINITIONS
gate1 = go.single_qubit_gate_to_full_gate(go.H(), qubit_amount, 1)
gate2 = go.single_qubit_gate_to_full_gate(go.H(), qubit_amount, 2)

#print(timeit.timeit("go.single_qubit_gate_to_full_gate(go.H(), 3, 1)", setup='import gate_operations as go'))








# SEQUENCE

print(state)
state = go.gate_operation(state, gate1)
print(state)
state = go.gate_operation(state, gate2)
print(state)
state = go.gate_operation(state, go.CNOT(qubit_amount,1,2))
print(state)
state = go.gate_operation(state, gate1)
print(state)
state = go.gate_operation(state, gate2)
print(state)



# SWAP From CNOTs

#SWAP_almost = np.matmul(go.CNOT(qubit_amount,2,1),go.CNOT(qubit_amount,1,2))
#SWAP = np.matmul(go.CNOT(qubit_amount,1,2),SWAP_almost)
#print(SWAP)


