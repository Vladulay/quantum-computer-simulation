# New FIle 
import numpy as np
import qutip as qt



def single_qubit_gate_operation (input_qubit: np.array, gate: np.array):
    # check if unitary
    if not unitary_check(gate):
        print("Error: Gate is not a unitary. Skipping gate.")
        return input_qubit
    
    # check shapes
    if input_qubit.shape[0] != gate.shape[0]:
        print("Error: Gate is not of same dimension as state. Skipping gate.")
        return input_qubit
        
   # multiply
    output_qubit = np.matmul(gate, input_qubit)

    return output_qubit


def unitary_check(matrix: np.array):
    shape = matrix.shape
    
    # check if square matrix
    if shape[0] != shape[1]:
        return False
    
    mult = np.matmul(matrix.conj().T, matrix)
    if not np.array_equal(mult, np.identity(2)):
        return False
    
    return True


def normalization_check(vector: np.array):
    length = np.linalg.norm(vector)
    if length != 1:
        print("State is not normalized. Correcting...")
        vector = vector/length
    return vector







# PIPELINE
def transform_to_bloch_vector(vector: np.array):
    vector[0].imag()










# EXPERIMENTAL AREA
out = single_qubit_gate_operation(np.array([1+2j,2]), np.array([[0,1],[1,0]]))
out = normalization_check(out)

print(out)
print(out[0].imag)

b = qt.Bloch()
#b.add_vectors(out)
b.show()



