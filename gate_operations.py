# New FIle 
import numpy as np
import qutip as qt
import timeit


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


def plot_bloch_state(state: np.array,  bloch_sphere: qt.Bloch, color: str = "royalblue"):
    colors = [color]
    
    bloch_sphere.add_vectors(spherical_to_cartesian(transform_to_bloch_vector(state)), colors)


def plotted_single_qubit_operation(in_state: np.array, gate: np.array, bloch_sphere: qt.Bloch, color_in: str, color_out: str):
    # plot of in state
    plot_bloch_state(in_state, bloch_sphere, color_in)

    out_state = gate_operation(in_state, gate)
    
    # plot out_state
    plot_bloch_state(out_state, bloch_sphere, color_out)
    
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


def single_qubit_gate_to_full_gate(gate: np.array ,qubit_amount: int, qubit_index: int): 
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
        new_gate = np.kron(single_qubit_gates.pop(), new_gate)
    
    
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


def CNOT(qubit_amount: int, control_qubit_index: int, target_qubit_index: int):
    """
    Constructs an arbitrary CNOT gate matrix representation

    Args:
        qubit_amount: amount of qubits in the system
        control_qubit_index: index of the control qubit
        target_qubit_index: index of the target qubit

    Returns:
        np.array: correct CNOT gate matrix, dimensions (qubit_amount x qubit_amount)
    """
    
    
    # Check for correct index stuctures
    if control_qubit_index == target_qubit_index:
        print("Error: target and control qubit cannot have same index. Skipping gate.")
        return np.identity(2**qubit_amount)
    
    if control_qubit_index > qubit_amount or target_qubit_index > qubit_amount:
        print("Error: target or control qubit cannot have greater index then qubit amount. Skipping gate.")
        return np.identity(2**qubit_amount)
      
    # Prepare rightmost matrices of tensor product
    leave_matrix = np.identity(2)
    flip_matrix = np.identity(2)
    
    if control_qubit_index == qubit_amount:
        leave_matrix = np.array([[1, 0],
                                 [0, 0]])  # |0><0|
        flip_matrix = np.array([[0, 0],
                                [0, 1]])  # |1><1|
    
    if target_qubit_index == qubit_amount:
        flip_matrix = X()
    
    # Compute tensor products
    index = qubit_amount - 1
    while index > 0:
        if index == target_qubit_index:
            leave_matrix = np.kron(np.identity(2),leave_matrix)
            flip_matrix = np.kron(X(),flip_matrix)
            
        elif index == control_qubit_index:
            leave_matrix = np.kron(np.array([[1, 0],[0, 0]]),leave_matrix) # apply |0><0|
            flip_matrix = np.kron(np.array([[0, 0],[0, 1]]),flip_matrix)   # apply |1><1|
        
        else:
            leave_matrix = np.kron(np.identity(2),leave_matrix)
            flip_matrix = np.kron(np.identity(2),flip_matrix)
        
        index -= 1
    
    CNOT = leave_matrix + flip_matrix # generalized version of CNOT=(∣0⟩⟨0∣⊗I)+(∣1⟩⟨1∣⊗X)
    return CNOT


def SWAP(qubit_amount: int, qubit1_index: int, qubit2_index: int):
    """
    Constructs an arbitrary SWAP gate matrix representation

    Args:
        qubit_amount: amount of qubits in the system
        qubit1_index, qubit2_index: indices of the qubits one wants to swap

    Returns:
        np.array: correct SWAP gate matrix, dimensions (qubit_amount x qubit_amount)
    """
    
    
    # Check for correct index stuctures
    if qubit1_index == qubit2_index:
        print("Error: Swaping qubits with same index is useless. Skipping gate.")
        return np.identity(2**qubit_amount)
    
    if qubit1_index > qubit_amount or qubit2_index > qubit_amount:
        print("Error: a swap qubit index is out of range. Skipping gate.")
        return np.identity(2**qubit_amount)
    
    # sort indices by magnitude 1>2
    if qubit1_index < qubit2_index:
        temp = qubit1_index
        qubit1_index = qubit2_index
        qubit2_index = temp
    
    # Prepare rightmost matrices of tensor product
    matrix_00 = np.identity(2)
    matrix_01 = np.identity(2)
    matrix_10 = np.identity(2)
    matrix_11 = np.identity(2)
    
    if qubit1_index == qubit_amount:
        matrix_00 = np.array([[1, 0],
                              [0, 0]])  # |0><0|
        matrix_01 = np.array([[0, 1],
                              [0, 0]])  # |0><1|
        matrix_10 = np.array([[0, 0],
                              [1, 0]])  # |1><0|
        matrix_11 = np.array([[0, 0],
                              [0, 1]])  # |1><1|
    
    # Compute tensor products
    index = qubit_amount - 1
    while index > 0:
        if index == qubit1_index:
            matrix_00 = np.kron(np.array([[1, 0],[0, 0]]),matrix_00)  # |0><0|
            matrix_01 = np.kron(np.array([[0, 1],[0, 0]]),matrix_01)  # |0><1|
            matrix_10 = np.kron(np.array([[0, 0],[1, 0]]),matrix_10)  # |1><0|
            matrix_11 = np.kron(np.array([[0, 0],[0, 1]]),matrix_11)  # |1><1|
            
        elif index == qubit2_index:
            matrix_00 = np.kron(np.array([[1, 0],[0, 0]]),matrix_00)  # |0><0|
            matrix_01 = np.kron(np.array([[0, 0],[1, 0]]),matrix_01)  # |1><0|
            matrix_10 = np.kron(np.array([[0, 1],[0, 0]]),matrix_10)  # |0><1|
            matrix_11 = np.kron(np.array([[0, 0],[0, 1]]),matrix_11)  # |1><1|
        
        else:
            matrix_00 = np.kron(np.identity(2),matrix_00)  
            matrix_01 = np.kron(np.identity(2),matrix_01)  
            matrix_10 = np.kron(np.identity(2),matrix_10)  
            matrix_11 = np.kron(np.identity(2),matrix_11)  
        
        index -= 1
    
    SWAP = matrix_00 + matrix_01 + matrix_10 + matrix_11
    return SWAP


def CNOT_from_SWAP(qubit_amount: int, control_index: int, target_index: int):
    """
    Constructs an arbitrary CNOT gate matrix representation by using SWAP gates and the CNOT gate w/ control 1 & target 2.

    Args:
        qubit_amount: amount of qubits in the system
        control_index: index of the control qubit
        target_index: index of the target qubit

    Returns:
        np.array: correct CNOT gate matrix, dimensions (qubit_amount x qubit_amount)
    """
    
    # Check for correct index stuctures
    if control_index == target_index:
        print("Error: target and control qubit cannot have same index. Skipping gate.")
        return np.identity(2**qubit_amount)
    
    if control_index > qubit_amount or target_index > qubit_amount:
        print("Error: target or control qubit cannot have greater index then qubit amount. Skipping gate.")
        return np.identity(2**qubit_amount)
        
    
    
    
    gate = SWAP(qubit_amount, 1, control_index)  # SWAP controls & targets to 1 & 2  
    if target_index != 1:      
        gate = np.matmul(SWAP(qubit_amount, 2, target_index) ,gate)
        
        
        gate = np.matmul(CNOT(qubit_amount,1,2) ,gate) # apply CNOT
        
        
        gate = np.matmul(SWAP(qubit_amount, 2, target_index) ,gate)
        gate = np.matmul(SWAP(qubit_amount, 1, control_index) ,gate)  # SWAP back
    else: 
        if control_index != 2:
            gate = np.matmul(SWAP(qubit_amount, 2, control_index) ,gate)
        
        gate = np.matmul(CNOT(qubit_amount,1,2) ,gate) # apply CNOT
        
        if control_index != 2:
            gate = np.matmul(SWAP(qubit_amount, 2, control_index) ,gate)
        
        gate = np.matmul(SWAP(qubit_amount, 1, control_index) ,gate) # SWAP back
    
    return gate









