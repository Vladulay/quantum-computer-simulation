# New FIle 
import numpy as np


# vector v and gate G
def single_qubit_gate_operation (input_qubit: np.array, gate: np.array):
    # check if unitary
    if not unitary_check(gate):
        print("Error: Gate is not a unitary. Skipping gate.")
        return input_qubit
    
    # check shapes
    if input_qubit.shape[0] != gate.shape[0]:
        print("Error: Gate is not of same dimension as state. Skipping gate.")
        return input_qubit
        
   # multiply  out = G*v
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
        #print("State is not normalized. Correcting...")
        vector = vector/length
    return vector


# (a,b) to (theta, phi, r)
def transform_to_bloch_vector(vector: np.array):
    angles = np.angle(vector)
    normalized_vector = normalization_check(vector)
    length_1 = np.sqrt(np.real(normalized_vector[0])**2 + np.imag(normalized_vector[0])**2)
    
    
    # cos(theta/2) = length_1 (?)
    theta = 2*np.arccos(length_1)
    phi = angles[1]-angles[0]
    r = np.linalg.norm(vector)
    
    bloch_vector = np.array([theta, phi, r])
    
    return bloch_vector


#(theta, phi, r) to (x,y,z)
def spherical_to_cartesian(vector_sph: np.array):
    x = vector_sph[2] * np.sin(vector_sph[0]) * np.cos(vector_sph[1])
    y = vector_sph[2] * np.sin(vector_sph[0]) * np.sin(vector_sph[1])
    z = vector_sph[2] * np.cos(vector_sph[0])
    
    cartesian_vector = np.array([x,y,z])
    
    return cartesian_vector






