# New FIle 
import numpy as np
#import qutip



def single_qubit_gate_operation (input_qubit: np.array, gate: np.array):
    # check if unitary
    if not _unitary_check(gate):
        return input_qubit
    
    # check shapes
    if input_qubit.shape[0] != gate.shape[0]:
        print("Error: Gate is not of same dimension as state. Skipping gate.")
        return input_qubit
    
    # normalize vector if it is not
    length = np.linalg.norm(input_qubit)
    if length != 1:
        print("State is not normalized. Correcting...")
        input_qubit = input_qubit/length
        
   # multiply
    output_qubit = np.matmul(gate, input_qubit)

    return output_qubit




def _unitary_check(matrix: np.array):
    shape = matrix.shape
    
    if shape[0] != shape[1]:
        print("Error: Gate is not a square matrix. Skipping gate.")
        return False
    
    mult = np.matmul(matrix.conj().T, matrix)
    if not np.array_equal(mult, np.identity(2)):
        print("Error: Gate is not unitary. Skipping gate.")
        return False
    
    return True






print(single_qubit_gate_operation(np.array([1,2]), np.array([[0,1],[1,0]])))




# plot

#b = qutip.Bloch()
#b.show()


