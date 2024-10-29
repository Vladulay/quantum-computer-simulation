# New FIle 
import numpy as np


# vector v and gate G
def single_qubit_gate_operation (input_qubit: np.array, gate: np.array):
    # check if unitary
    if not unitary_check(gate):
        print("Error: Gate " + str(gate) + " is not a unitary. Skipping gate.")
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
    
    compare = np.isclose(mult, np.identity(2))
    if False in compare:
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






















# GATES

# Pauli-X
def X():
    return np.array([[0,1],[1,0]])


# Pauli-Y
def Y():
    return np.array([[0,-1j],[1j,0]])


# Pauli-Z
def Z():
    return np.array([[1,0],[0,-1]])


# Hadamard
def H():
    return np.array([[1,1],[1,-1]]) * (1/np.sqrt(2))


# Phase
def S():
    return np.array([[1,0],[0,1j]])


# PI/8
def T():
    return np.array([[1,0],[0,np.exp(1j*np.pi / 4)]])


# general x rotation
def R_x(angle: float):
    return np.cos(angle/2) * np.identity(2) - 1j * np.sin(angle/2) * X()


# general y rotation
def R_y(angle: float):
    return np.cos(angle/2) * np.identity(2) - 1j * np.sin(angle/2) * Y()


# general z rotation
def R_z(angle: float):
    return np.cos(angle/2) * np.identity(2) - 1j * np.sin(angle/2) * Z()


# general rotation
def R(direction: np.array ,angle: float):
    print(direction.shape)
    if direction.shape != (3,):
        print("Rotation direction " + str(direction) +  " is not a valid vector. Skipping rotation.")
        return np.identity(2)
    
    direction = normalization_check(direction)
    return np.cos(angle/2) * np.identity(2) - 1j * np.sin(angle/2) *( direction[0] * X() + direction[1] * Y() + direction[2] * Z())