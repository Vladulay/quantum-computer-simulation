# New FIle 
import numpy as np
import qutip as qt


# MAIN FUNCTIONS

# vector v and gate G
def gate_operation (input_state: np.array, gate: np.array):
    # check if unitary
    if not unitary_check(gate):
        print("Error: Gate " + str(gate) + " is not a unitary. Skipping gate.")
        return input_state
    
        
    # multiply  out = G*v
    output_qubit = np.matmul(gate, input_state)

    return output_qubit


def plotted_single_qubit_operation(in_state: np.array, gate: np.array, bloch_sphere: qt.Bloch):
    # plot of in state
    bloch_sphere.add_vectors(spherical_to_cartesian(transform_to_bloch_vector(in_state)))

    out_state = gate_operation(in_state, gate)
    
    # plot out_state
    bloch_sphere.add_vectors(spherical_to_cartesian(transform_to_bloch_vector(out_state)))
    
    return out_state










# HELPERS

def unitary_check(matrix: np.array):
    shape = matrix.shape
    
    # check if square matrix
    if shape[0] != shape[1]:
        return False
    
    mult = np.matmul(matrix.conj().T, matrix)
    
    compare = np.isclose(mult, np.identity(matrix.shape[0]))
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


def tensor_states(state1: np.array, state2: np.array):
    state_length = state1.shape[0]*state2.shape[0]
    
    new_state = np.zeros((state_length,))
    
    state_index = 0
    state1_index = 0
    while state1_index < state1.shape[0]:
        state2_index = 0
        while state2_index < state2.shape[0]:
            
            new_state[state_index] = state1[state1_index] * state2[state2_index]
            
            state2_index += 1
            state_index += 1
        state1_index += 1

        

    return new_state


def tensor_gates(gate1: np.array, gate2: np.array):
    gate_dim = gate1.shape[0] * gate2.shape[0]
    
    new_gate = np.zeros((gate_dim,gate_dim),complex)
    
    gate1_lin_index = 0
    while gate1_lin_index < gate1.shape[0]:
        
        gate1_col_index = 0
        while gate1_col_index < gate1.shape[0]:
            
            gate2_lin_index = 0
            while gate2_lin_index < gate2.shape[0]:
                
                gate2_col_index = 0
                while gate2_col_index < gate2.shape[0]:
                    
                    
                    new_gate_col_index = gate1_col_index * gate2.shape[0] + gate2_col_index
                    new_gate_lin_index = gate1_lin_index * gate2.shape[0] + gate2_lin_index
                    
                    new_gate[new_gate_lin_index , new_gate_col_index] = gate1[gate1_lin_index, gate1_col_index] * gate2[gate2_lin_index, gate2_col_index]
                    
                    
                    gate2_col_index += 1
                
                gate2_lin_index += 1
            
            gate1_col_index += 1
        
        gate1_lin_index += 1
    
    return new_gate



def single_qubit_gate_to_full_gate(gate: np.array ,qubit_amount: int, qubit_index: int): # WRONG
    single_qubit_gates = []
    index = 0
    while index < qubit_amount:
        if index == qubit_index - 1:
            single_qubit_gates.append(gate)
        else:
            single_qubit_gates.append(np.identity(2,complex))
        
        index += 1
    
    new_gate = single_qubit_gates.pop()
    
    while len(single_qubit_gates) > 0:
        new_gate = tensor_gates(single_qubit_gates.pop(), new_gate)
    
    
    return new_gate













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
    if direction.shape != (3,):
        print("Rotation direction " + str(direction) +  " is not a valid vector. Skipping rotation.")
        return np.identity(2)
    
    direction = normalization_check(direction)
    return np.cos(angle/2) * np.identity(2) - 1j * np.sin(angle/2) *( direction[0] * X() + direction[1] * Y() + direction[2] * Z())


# CURRENTLY ALWAYS 1ST QUBIT CONTROL AND 2ND TARGET
def CNOT():
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])