# Hi
import gate_operations as go
import numpy as np
import qutip as qt

# GOALS
# DONE: Extend and play around more with the Bloch sphere visualizer by implementing all the standard gates from NS.
# DONE: Implement the gate from eq. (4.8) in NS, convince yourself that it implements the rotation about the axis defined by n.
# DONE: Create a framework for single qubit circuit creation.
# DONE: Extend the circuit framework to 2 qubits, where we can apply any one- and two-qubit gates between the qubits.
# - Look into/implement some automated testing of your functions inside python (see e.g. Unittest). 










# INPUT SPACE
qubit_amount = 3 # all qubits are initilized as |0〉









# state prep |0000...0000〉
state = np.zeros((2**qubit_amount,))
state[0] = 1




# GATE DEFINITIONS
gate1 = go.single_qubit_gate_to_full_gate(go.H(), qubit_amount, 1)
gate2 = go.single_qubit_gate_to_full_gate(go.H(), qubit_amount, 2)








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



print(go.CNOT(qubit_amount,2,1))



