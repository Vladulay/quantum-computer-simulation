import numpy as np
import qutip as qt
import random
from scipy.sparse import csr_matrix
from scipy.sparse import kron
import itertools
from functools import reduce

import matplotlib.pyplot as plt

sparse_advantage = 9


# MAIN FUNCTIONS

# vector v and gate G
def gate_operation (input_state, gate, qubit_amount: int = 1):
    """
    Computes a gate multiplication. Takes in either both as 'np.ndarray' or
    both as 'scipy.sparse.csr_matrix'

        input_state: state to which to apply the gate
        gate: gate which to apply

    Returns:
        np.array / scipy.sparse.csr_matrix: state after gate application
    """
    
    # dense matrices
    if type(input_state) == np.ndarray and type(gate) == np.ndarray:
    
        return _dense_gate_operation(input_state, gate, qubit_amount) 
    
    # sparse matrices
    elif type(input_state) == csr_matrix and type(gate) == csr_matrix:
        
        output_state = None
        
        # pure state 
        if len(input_state.shape) == 1 or 1 in input_state.shape:
            output_state  = gate @ input_state.reshape(-1, 1)
            
        # mixed state (NOT WORKING YET)
        else:
            output_state  = gate @ input_state @ gate.conj().T
    
        return output_state
    
    # unsupported type
    else:
        print("Error: unsupported typing in gate operation.")
        
        return input_state


def _dense_gate_operation (input_state, gate, qubit_amount: int = 1):
    # pure states
    if len(input_state.shape) == 1 or 1 in input_state.shape:
        return _pure_gate_operation(input_state, gate, qubit_amount)
    
    # mixed states
    else: 
        return _mixed_gate_operation(input_state, gate, qubit_amount)


def _pure_gate_operation (input_state, gate, qubit_amount: int = 1):
    if qubit_amount >= sparse_advantage:
        gate = csr_matrix(gate)     
        input_state = csr_matrix(input_state)
        
        
        # multiply  out = G*v
        output_qubit  = gate @ input_state.reshape(-1, 1)
            
        output_qubit = output_qubit.toarray().T
            
        return output_qubit 
    
    # check if unitary
    if not unitary_check(gate):
        print("Error: Gate " + str(gate) + " is not a unitary. Skipping gate.")
        return input_state
        
    # multiply  out = G*v
    output_state = np.matmul(gate, input_state)
    
    return output_state


def _mixed_gate_operation (input_state, gate, qubit_amount: int = 1):
    # check if unitary
    if not unitary_check(gate):
        print("Error: Gate " + str(gate) + " is not a unitary. Skipping gate.")
        return input_state
        
    # multiply  out = G*v
    output_state = gate @ input_state @ gate.conj().T
    
    return output_state


def plot_bloch_state(state: np.array,  bloch_sphere: qt.Bloch, color: str = "royalblue"):
    colors = [color]
    
    # pure state
    if state.shape == (2,):
        bloch_sphere.add_vectors(spherical_to_cartesian(transform_to_bloch_vector(state)), colors)
        
    # mixed state
    else:
        # mixed state
        x = state[0,1] + state[1,0]
        y = (state[0,1] - state[1,0]) / 1j
        z = 2 *state[0,0] - 1
        
        bloch_sphere_vector = np.array([x,y,z])
        
        bloch_sphere.add_vectors(bloch_sphere_vector, colors)
        

def plotted_single_qubit_operation(in_state: np.array, gate: np.array, bloch_sphere: qt.Bloch, color_in: str, color_out: str):
    # plot of in state
    plot_bloch_state(in_state, bloch_sphere, color_in)

    out_state = gate_operation(in_state, gate)
    
    # plot out_state
    plot_bloch_state(out_state, bloch_sphere, color_out)
    
    return out_state









# HELPERS

def unitary_check(matrix: np.array):
    """
    Check a np.ndarray for unitarity

        matrix: np.ndarray

    Returns:
        bool
    """
    
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
    """
    Normalizes a pure state

        vector: np.ndarray state

    Returns:
        np.ndarray: normalized state
    """
    
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


def single_qubit_gate_to_full_gate(gate ,qubit_amount: int, qubit_index: int): 
    single_qubit_gates = []
    index = 0
    while index < qubit_amount:
        if index == qubit_index - 1:
            single_qubit_gates.append(gate)
        else:
            single_qubit_gates.append(np.identity(2,complex))
        
        index += 1
    
    if type(gate) == np.ndarray:
        
        #if qubit_amount >= 8:
        
        new_gate = csr_matrix(single_qubit_gates.pop())
        
        while len(single_qubit_gates) > 0:
            new_gate = kron(csr_matrix(single_qubit_gates.pop()) , new_gate)
            
        new_gate = csr_matrix(new_gate).toarray()
            
        return new_gate
        
        # new_gate = single_qubit_gates.pop()
        
        # while len(single_qubit_gates) > 0:
        #     new_gate = np.kron(single_qubit_gates.pop(), new_gate)
        
        
        # return new_gate
    
    elif type(gate) == csr_matrix:
        
        new_gate = csr_matrix(single_qubit_gates.pop())
    
        while len(single_qubit_gates) > 0:
            new_gate = kron(csr_matrix(single_qubit_gates.pop()) , new_gate)
        
        new_gate = csr_matrix(new_gate)
        
        return new_gate
    
    else:
        print("Error: unsupported typing in gate operation.")
        
        return gate
    
    
def multi_single_qubit_gates_to_full_gate(gate ,qubit_amount: int, qubit_indices: []): 
    single_qubit_gates = []
    
    index = 0
    while index < qubit_amount:
        if index + 1 in qubit_indices:
            single_qubit_gates.append(gate)
        else:
            single_qubit_gates.append(np.identity(2,complex))
        
        index += 1
    
    if type(gate) == np.ndarray:
        # if qubit_amount >= sparse_advantage:
        
        #     new_gate = csr_matrix(single_qubit_gates.pop())
            
        #     while len(single_qubit_gates) > 0:
        #         new_gate = kron(csr_matrix(single_qubit_gates.pop()) , new_gate)
                
        #     new_gate = csr_matrix(new_gate)
                
        #     new_gate = csr_matrix(new_gate).toarray()
                
        #     return new_gate
        
        new_gate = single_qubit_gates.pop()
        
        while len(single_qubit_gates) > 0:
            new_gate = np.kron(single_qubit_gates.pop(), new_gate)

        return new_gate
    
    elif type(gate) == csr_matrix:
        new_gate = csr_matrix(single_qubit_gates.pop())
    
        while len(single_qubit_gates) > 0:
            new_gate = kron(csr_matrix(single_qubit_gates.pop()) , new_gate)
        
        new_gate = csr_matrix(new_gate)
        
        return new_gate
    
    else:
        print("Error: unsupported typing in gate operation.")
        
        return gate
    











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


def CNOT_csr(qubit_amount: int, control_qubit_index: int, target_qubit_index: int):
    """
    Constructs an arbitrary CNOT gate matrix representation

        qubit_amount: amount of qubits in the system
        control_qubit_index: index of the control qubit
        target_qubit_index: index of the target qubit

    Returns:
        scipy csr_matrix: correct CNOT gate matrix, dimensions (qubit_amount x qubit_amount)
    """
    
    # Check for correct index stuctures
    if control_qubit_index == target_qubit_index:
        print("Error: target and control qubit cannot have same index. Skipping gate.")
        return csr_matrix(np.identity(2**qubit_amount))
    
    if control_qubit_index > qubit_amount or target_qubit_index > qubit_amount:
        print("Error: target or control qubit cannot have greater index then qubit amount. Skipping gate.")
        return csr_matrix(np.identity(2**qubit_amount))
    
    # Prepare rightmost matrices of tensor product
    leave_matrix = csr_matrix(np.identity(2))
    flip_matrix = csr_matrix(np.identity(2))
    
    if control_qubit_index == qubit_amount:
        leave_matrix = csr_matrix((np.array([1]), (np.array([0]), np.array([0]))), shape=(2, 2))  # |0><0|
        flip_matrix = csr_matrix((np.array([1]), (np.array([1]), np.array([1]))), shape=(2, 2))  # |1><1|
    
    if target_qubit_index == qubit_amount:
        flip_matrix = csr_matrix(X())
    
    # Compute tensor products
    index = qubit_amount - 1
    while index > 0:
        if index == target_qubit_index:
            leave_matrix = kron(csr_matrix(np.identity(2)),leave_matrix)
            flip_matrix = kron(csr_matrix(X()) , flip_matrix)
            
        elif index == control_qubit_index:
            leave_matrix = kron(csr_matrix((np.array([1]), (np.array([0]), np.array([0]))), shape=(2, 2)) , leave_matrix) # apply |0><0|
            flip_matrix = kron(csr_matrix((np.array([1]), (np.array([1]), np.array([1]))), shape=(2, 2)) , flip_matrix)   # apply |1><1|
        
        else:
            leave_matrix = kron(csr_matrix(np.identity(2)) , leave_matrix)
            flip_matrix = kron(csr_matrix(np.identity(2)) , flip_matrix)
        
        index -= 1
    
    CNOT = leave_matrix + flip_matrix # generalized version of CNOT=(∣0⟩⟨0∣⊗I)+(∣1⟩⟨1∣⊗X)
    
    CNOT = csr_matrix(CNOT)
    
    return CNOT


def SWAP_csr(qubit_amount: int, qubit1_index: int, qubit2_index: int):
    """
    Constructs an arbitrary SWAP gate matrix representation

        qubit_amount: amount of qubits in the system
        qubit1_index, qubit2_index: indices of the qubits one wants to swap

    Returns:
        scipy csr_matrix: correct SWAP gate matrix, dimensions (qubit_amount x qubit_amount)
    """
    
    
    # Check for correct index stuctures
    if qubit1_index == qubit2_index:
        print("Warning: Swaping qubits with same index is useless. Skipping gate.")
        return csr_matrix(np.identity(2**qubit_amount)) 
    
    if qubit1_index > qubit_amount or qubit2_index > qubit_amount:
        print("Error: a swap qubit index is out of range. Skipping gate.")
        return csr_matrix(np.identity(2**qubit_amount)) 
    
    # sort indices by magnitude 1>2
    if qubit1_index < qubit2_index:
        temp = qubit1_index
        qubit1_index = qubit2_index
        qubit2_index = temp
    
    # Prepare rightmost matrices of tensor product
    matrix_00 = csr_matrix(np.identity(2)) 
    matrix_01 = csr_matrix(np.identity(2)) 
    matrix_10 = csr_matrix(np.identity(2)) 
    matrix_11 = csr_matrix(np.identity(2)) 
    
    if qubit1_index == qubit_amount:
        matrix_00 = csr_matrix((np.array([1]), (np.array([0]), np.array([0]))), shape=(2, 2))  # |0><0|
        matrix_01 = csr_matrix((np.array([1]), (np.array([0]), np.array([1]))), shape=(2, 2))  # |0><1|
        matrix_10 = csr_matrix((np.array([1]), (np.array([1]), np.array([0]))), shape=(2, 2))  # |1><0|
        matrix_11 = csr_matrix((np.array([1]), (np.array([1]), np.array([1]))), shape=(2, 2))  # |1><1|
    
    # Compute tensor products
    index = qubit_amount - 1
    while index > 0:
        if index == qubit1_index:
            matrix_00 = kron(csr_matrix((np.array([1]), (np.array([0]), np.array([0]))), shape=(2, 2)) , matrix_00)  # |0><0|
            matrix_01 = kron(csr_matrix((np.array([1]), (np.array([0]), np.array([1]))), shape=(2, 2)) , matrix_01)  # |0><1|
            matrix_10 = kron(csr_matrix((np.array([1]), (np.array([1]), np.array([0]))), shape=(2, 2)) , matrix_10)  # |1><0|
            matrix_11 = kron(csr_matrix((np.array([1]), (np.array([1]), np.array([1]))), shape=(2, 2)) , matrix_11)  # |1><1|
            
        elif index == qubit2_index:
            matrix_00 = kron(csr_matrix((np.array([1]), (np.array([0]), np.array([0]))), shape=(2, 2)) , matrix_00)  # |0><0|
            matrix_01 = kron(csr_matrix((np.array([1]), (np.array([1]), np.array([0]))), shape=(2, 2)) , matrix_01)  # |1><0|
            matrix_10 = kron(csr_matrix((np.array([1]), (np.array([0]), np.array([1]))), shape=(2, 2)) , matrix_10)  # |0><1|
            matrix_11 = kron(csr_matrix((np.array([1]), (np.array([1]), np.array([1]))), shape=(2, 2)) , matrix_11)  # |1><1|
        
        else:
            matrix_00 = kron(csr_matrix(np.identity(2)) , matrix_00)
            matrix_01 = kron(csr_matrix(np.identity(2)) , matrix_01)
            matrix_10 = kron(csr_matrix(np.identity(2)) , matrix_10)
            matrix_11 = kron(csr_matrix(np.identity(2)) , matrix_11) 
        
        index -= 1
    
    SWAP = matrix_00 + matrix_01 + matrix_10 + matrix_11
    
    SWAP = csr_matrix(SWAP)
    
    return SWAP


def CNOT(qubit_amount: int, control_qubit_index: int, target_qubit_index: int):
    """
    Constructs an arbitrary CNOT gate matrix representation

        qubit_amount: amount of qubits in the system
        control_qubit_index: index of the control qubit
        target_qubit_index: index of the target qubit

    Returns:
        np.array: correct CNOT gate matrix, dimensions (qubit_amount x qubit_amount)
    """
    
    # sparse advantage regime
    if qubit_amount >= sparse_advantage:
        CNOT = CNOT_csr(qubit_amount, control_qubit_index, target_qubit_index).toarray()
        return CNOT
    
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

        qubit_amount: amount of qubits in the system
        qubit1_index, qubit2_index: indices of the qubits one wants to swap

    Returns:
        np.array: correct SWAP gate matrix, dimensions (qubit_amount x qubit_amount)
    """
    
    # sparse advantage regime
    if qubit_amount >= sparse_advantage:
        SWAP = SWAP_csr(qubit_amount, qubit1_index, qubit2_index).toarray()
        return SWAP
    
    # Check for correct index stuctures
    if qubit1_index == qubit2_index:
        print("Warning: Swaping qubits with same index is useless. Skipping gate.")
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






# Instruction syntax

class instruction:
    def __init__(self, gate=None, qubit=None, angle=None, direction=None, probability=None):
        if gate is None:
            gate = ""
        self.gate = gate
        
        if qubit is None:
            qubit = [1]
        self.qubit = qubit
        
        if direction is None:
            direction = np.array([0,0,1])
        self.direction = direction
        
        if angle is None:
            angle = 0.0
        self.angle = angle
        
        if probability is None:
            probability = 1.0
        self.probability = probability
    
    
    def create_from_list(instruction_list=None):
        if instruction_list is None:
            print("Warning: No instruction list given")
            return
        
        gate = instruction_list[0]
        qubit = instruction_list[1]
        
        list_size = len(instruction_list)
        
        if gate == "bitflip" or gate == "phaseflip" or gate == "ampdamp" or gate == "depol":
            probability = instruction_list[1]
            #qubit = instruction_list[2]
            self = instruction(gate,1, None, None, probability)
            
            if gate == "bitflip" or gate == "phaseflip" or gate == "ampdamp":
                self = instruction(gate,instruction_list[2], None, None, probability)
            return self
        
        if list_size == 3:
            angle = instruction_list[2]
            self = instruction(gate,qubit,angle)
            return self 
        
        if list_size == 4:
            angle = instruction_list[2]
            direction = instruction_list[3]
            self = instruction(gate,qubit,angle, direction)
            return self
        
        self = instruction(gate,qubit)
        return self
    
    
    def random_instruction(qubit_amount: int = 1):
        choice = random.randint(0,11)
        
        match choice:
            case 0:
                self = instruction("H",[random.randint(1,qubit_amount)])
            case 1:
                self = instruction("X",[random.randint(1,qubit_amount)])
            case 2:
                self = instruction("Y",[random.randint(1,qubit_amount)])
            case 3:
                self = instruction("Z",[random.randint(1,qubit_amount)])
            case 4:
                self = instruction("S",[random.randint(1,qubit_amount)])
            case 5:
                self = instruction("T",[random.randint(1,qubit_amount)])
            case 6:
                self = instruction("Rx",[random.randint(1,qubit_amount)], random.random()*2*np.pi)
            case 7:
                self = instruction("Ry",[random.randint(1,qubit_amount)], random.random()*2*np.pi)
            case 8:
                self = instruction("Rz",[random.randint(1,qubit_amount)], random.random()*2*np.pi)
            case 9:
                self = instruction("R",[random.randint(1,qubit_amount)], random.random()*2*np.pi, np.array([random.random()-0.5,random.random()-0.5,random.random()-0.5]))
            case 10:
                if qubit_amount < 2:
                    return instruction.random_instruction()
                
                control_qubit = random.randint(1,qubit_amount)
                target_qubit = control_qubit
                while target_qubit == control_qubit:
                    target_qubit = random.randint(1,qubit_amount)
                
                self = instruction("CNOT",[control_qubit,target_qubit])
            case 11:
                if qubit_amount < 2:
                    return instruction.random_instruction()
                
                qubit1 = random.randint(1,qubit_amount)
                qubit2 = qubit1
                while qubit1 == qubit2:
                    qubit2 = random.randint(1,qubit_amount)
                
                self = instruction("SWAP",[qubit1,qubit2])
             
        return self
    
    
    def random_single_qubit_instruction(qubit_amount: int = 1):
        choice = random.randint(0,9)
        
        match choice:
            case 0:
                self = instruction("H",[random.randint(1,qubit_amount)])
            case 1:
                self = instruction("X",[random.randint(1,qubit_amount)])
            case 2:
                self = instruction("Y",[random.randint(1,qubit_amount)])
            case 3:
                self = instruction("Z",[random.randint(1,qubit_amount)])
            case 4:
                self = instruction("S",[random.randint(1,qubit_amount)])
            case 5:
                self = instruction("T",[random.randint(1,qubit_amount)])
            case 6:
                self = instruction("Rx",[random.randint(1,qubit_amount)], random.random()*2*np.pi)
            case 7:
                self = instruction("Ry",[random.randint(1,qubit_amount)], random.random()*2*np.pi)
            case 8:
                self = instruction("Rz",[random.randint(1,qubit_amount)], random.random()*2*np.pi)
            case 9:
                self = instruction("R",[random.randint(1,qubit_amount)], random.random()*2*np.pi, np.array([random.random()-0.5,random.random()-0.5,random.random()-0.5]))
             
        return self
    
    
    def chunked_random_instruction(qubit_amount: int = 1, chunk_size: int = 3):
        choice = random.randint(0,11)
        
        match choice:
            case 0:
                self = instruction("H",[random.randint(1,qubit_amount)])
            case 1:
                self = instruction("X",[random.randint(1,qubit_amount)])
            case 2:
                self = instruction("Y",[random.randint(1,qubit_amount)])
            case 3:
                self = instruction("Z",[random.randint(1,qubit_amount)])
            case 4:
                self = instruction("S",[random.randint(1,qubit_amount)])
            case 5:
                self = instruction("T",[random.randint(1,qubit_amount)])
            case 6:
                self = instruction("Rx",[random.randint(1,qubit_amount)], random.random()*2*np.pi)
            case 7:
                self = instruction("Ry",[random.randint(1,qubit_amount)], random.random()*2*np.pi)
            case 8:
                self = instruction("Rz",[random.randint(1,qubit_amount)], random.random()*2*np.pi)
            case 9:
                self = instruction("R",[random.randint(1,qubit_amount)], random.random()*2*np.pi, np.array([random.random()-0.5,random.random()-0.5,random.random()-0.5]))
            case 10:
                if qubit_amount < 2:
                    return instruction.random_instruction()
                
                control_qubit = random.randint(1, qubit_amount)
                
                base = control_qubit - control_qubit % chunk_size
                
                if base == control_qubit and base != 1:
                    base = base - chunk_size
                
                target_qubit = control_qubit
                if base + 1 != qubit_amount:
                    while target_qubit == control_qubit or target_qubit > qubit_amount:
                        target_qubit = base + random.randint(1, chunk_size)
                
                if target_qubit == control_qubit:
                    return instruction.random_instruction()
                
                self = instruction("CNOT",[control_qubit,target_qubit])
            case 11:
                if qubit_amount < 2:
                    return instruction.random_instruction()
                
                qubit1 = random.randint(1, qubit_amount)

                base = qubit1 - qubit1 % chunk_size

                if base == qubit1 and base != 1:
                    base = base - chunk_size

                qubit2 = qubit1
                if base + 1 != qubit_amount:
                    while qubit2 == qubit1 or qubit2 > qubit_amount:
                        qubit2 = base + random.randint(1, chunk_size)
                
                if qubit1 == qubit2:
                    return instruction.random_instruction()
                
                self = instruction("SWAP",[qubit1,qubit2])
             
        return self
    
     

def apply_instruction(state ,instruction: instruction(), qubit_amount: int = None):
    """
    Applies a gate to a state, defined by an instruction class object.

        state: input state
        instruction: instruction class object defining gate operation
        qubit_amount: amount of qubits in the system

    Returns:
        np.array / scipy.csr_matrix: state after application
    """
    
    if qubit_amount is None:
        if len(state.shape) == 1:
            qubit_amount = round(np.log2(len(state)))
        else:
            #print(state[0])
            qubit_amount = round(np.log2(len(state[0])))
            
        
        
    
    if type(state) == np.ndarray:
        match instruction.gate:
            case "H":
                state = gate_operation(state, multi_single_qubit_gates_to_full_gate(H(), qubit_amount, instruction.qubit), qubit_amount)
                return state
            case "X":
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(X(), qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "Y":
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(Y(), qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "Z":
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(Z(), qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "S":
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(S(), qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "T":
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(T(), qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "Rx":
                 gate = R_x(instruction.angle)
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(gate, qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "Ry":
                 gate = R_y(instruction.angle)
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(gate, qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "Rz":
                 gate = R_z(instruction.angle)
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(gate, qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "R":
                 gate = R(instruction.direction, instruction.angle)
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(gate, qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "CNOT":
                 gate = CNOT(qubit_amount, instruction.qubit[0], instruction.qubit[1])
                 state = gate_operation(state, gate, qubit_amount)
                 return state
            case "SWAP":
                 gate = SWAP(qubit_amount, instruction.qubit[0], instruction.qubit[1])
                 state = gate_operation(state, gate, qubit_amount)
                 return state
            case "bitflip":
                 state = bit_flip_channel(state, instruction.probability, instruction.qubit[0])
                 return state
            case "phaseflip":
                 state = phase_flip_channel(state, instruction.probability, instruction.qubit[0])
                 return state
            case "ampdamp":
                 state = amplitude_damping_channel(state, instruction.probability, instruction.qubit[0])
                 return state
            case "depol":
                 state = depolarizing_channel(state, qubit_amount, instruction.probability)
                 return state
            case _:
                 return state
    
    elif type(state) == csr_matrix:
        match instruction.gate:
            case "H":
                state = gate_operation(state, multi_single_qubit_gates_to_full_gate(csr_matrix(H()) , qubit_amount, instruction.qubit), qubit_amount)
                return state
            case "X":
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(csr_matrix(X()), qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "Y":
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(csr_matrix(Y()), qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "Z":
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(csr_matrix(Z()), qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "S":
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(csr_matrix(S()), qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "T":
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(csr_matrix(T()), qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "Rx":
                 gate = csr_matrix(R_x(instruction.angle))
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(gate, qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "Ry":
                 gate = csr_matrix(R_y(instruction.angle))
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(gate, qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "Rz":
                 gate = csr_matrix(R_z(instruction.angle))
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(gate, qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "R":
                 gate = csr_matrix(R(instruction.direction, instruction.angle))
                 state = gate_operation(state, multi_single_qubit_gates_to_full_gate(gate, qubit_amount, instruction.qubit), qubit_amount)
                 return state
            case "CNOT":
                 gate = CNOT_csr(qubit_amount, instruction.qubit[0], instruction.qubit[1])
                 state = gate_operation(state, gate, qubit_amount)
                 return state
            case "SWAP":
                 gate = SWAP_csr(qubit_amount, instruction.qubit[0], instruction.qubit[1])
                 state = gate_operation(state, gate, qubit_amount)
                 return state
            case _:
                 return state
    
    else:
        print("Error: unsupported typing in gate operation.")
        
        return state
    
    


def create_instruction_list(instruction_list: list):
    """
    Transforms a list of instruction lists (["R",[2],np.pi,np.array([1,0,0])]) into a list of instruction class objects
    
        instruction_list: list of instruction lists
        
    Returns:
        list: list of instruction objects
    """
    
    new_list = []
        
    for instr in instruction_list:
        new_list.append(instruction.create_from_list(instr))
        
    return new_list









# Projectors

def create_multi_qubit_projector_list(qubit_amount, P1, P2):
    """
    Generate all possible tensor products for matrices P1 and P2 based on bit patterns of length n.
    
    Parameters:
        P1 (numpy.ndarray): First matrix.
        P2 (numpy.ndarray): Second matrix.
        n (int): Length of bit patterns.
    
    Returns:
        dict: A dictionary where keys are bit patterns (as strings) and values are the resulting tensor products.
    """
    # Generate all bit patterns of length n
    bit_patterns = list(itertools.product([0, 1], repeat=qubit_amount))
    result = []
    
    for bits in bit_patterns:
        # Start with the first matrix in the tensor product
        tensor_product = P1 if bits[0] == 0 else P2
        # Iterate through the remaining bits to compute the tensor product
        for bit in bits[1:]:
            tensor_product = np.kron(tensor_product, P1 if bit == 0 else P2)
        # Store the result
        result.append(tensor_product)
    
    return result


def P_x(qubit_amount: int = 1):
    P_plus_x = np.array([[1, 1], [1, 1]]) / 2  # |+><+|
    P_minus_x = np.array([[1, -1], [-1, 1]]) / 2  # |-><-|
    
    return create_multi_qubit_projector_list(qubit_amount, P_plus_x, P_minus_x)


def P_y(qubit_amount: int = 1):
    P_plus_y = np.array([[1, -1j], [1j, 1]]) / 2  # |i+><i+|
    P_minus_y = np.array([[1, 1j], [-1j, 1]]) / 2  # |i-><i-|
    
    return create_multi_qubit_projector_list(qubit_amount, P_plus_y, P_minus_y)


def P_z(qubit_amount: int = 1):
    P_0 = np.array([[0, 0], [0, 1]])  # |1><1|
    P_1 = np.array([[1, 0], [0, 0]])  # |0><0|
    
    return create_multi_qubit_projector_list(qubit_amount, P_1, P_0)











# Measurement
def measure_projective(state, qubit_amount: int = 1, num_measurements: int = 1, projectors: list = None, return_frequencies: bool = True):
    """
    Perform measurements using a set of projectors.
    
    Args:
        state (np.ndarray): Quantum state to measure.
        qubit_amount (int): amount of qubits of the state
        num_measurements (int): Number of measurements.
        projectors (list of np.ndarray): List of projectors {P_i}.
        return_frequencies (bool): Whether to return frequencies or raw outcomes.
    
    Returns:
        dict: Frequencies of outcomes or amount of outcomes.
    """
    if projectors is None:
        measure_computational(state,qubit_amount,num_measurements,return_frequencies)
        
    
    # Normalize state
    state = state / np.linalg.norm(state)
    
    # Compute probabilities
    # pure states
    if len(state.shape) == 1 or 1 in state.shape:
        probabilities = [max (np.real(np.vdot(state, P @ state)),0) for P in projectors]
    # mixed states
    else:
        probabilities = [max (np.real(np.trace((P @ state))),0) for P in projectors]
    
    # Simulate measurement outcomes
    outcomes = np.random.choice(len(projectors), size=num_measurements, p=probabilities)
    
    if return_frequencies:
        frequencies = {i: np.sum(outcomes == i) / num_measurements for i in range(len(projectors))}
        return frequencies
    else:
        measurements = {i: np.sum(outcomes == i) for i in range(len(projectors))}
        return measurements


def measure_computational(state, qubit_amount: int = 1, num_measurements: int = 1, return_frequencies: bool = True):
    """
    Perform measurements in the computational basis (Z-basis).
    
    Args:
        state (np.ndarray): Quantum state to measure.
        qubit_amount (int): amount of qubits of the state
        num_measurements (int): Number of measurements.
        return_frequencies (bool): Whether to return frequencies or raw outcomes.
    
    Returns:
        dict: Frequencies of outcomes or amount of outcomes.
    """    
    # Define computational basis projectors
    projectors = P_z(qubit_amount)
    
    # Normalize state
    #state = state / np.linalg.norm(state)
    
    # Compute probabilities
    # pure states
    if len(state.shape) == 1 or 1 in state.shape:
        probabilities = [max (np.real(np.vdot(state, P @ state)),0) for P in projectors]
    # density matrix
    else:
        probabilities = [max (np.real(np.trace((P @ state))) ,0) for P in projectors]
        
        #print(probabilities)
    
    # Simulate measurement outcomes
    outcomes = np.random.choice(len(projectors), size=num_measurements, p=probabilities)
    
    if return_frequencies:
        frequencies = {i: np.sum(outcomes == i) / num_measurements for i in range(len(projectors))}
        
        # Find the maximum key to determine the number of bits required
        max_key = max(frequencies.keys())
        num_bits = max_key.bit_length()

        # Convert keys to bitstrings and create a new dictionary
        bitstring_dict = {format(key, f'0{num_bits}b'): value for key, value in frequencies.items()}
        
        return bitstring_dict
    else:
        measurements = {i: np.sum(outcomes == i) for i in range(len(projectors))}
        
        # Find the maximum key to determine the number of bits required
        max_key = max(measurements.keys())
        num_bits = max_key.bit_length()

        # Convert keys to bitstrings and create a new dictionary
        bitstring_dict = {format(key, f'0{num_bits}b'): value for key, value in measurements.items()}
        
        return bitstring_dict


def plot_measurement(measurement_dict, title="Measurement Outcomes", y_label = "frequencies / measurement amount",x_label = "measurement outcomes"):
    """
    Plot the frequencies of measurement outcomes as a bar plot.
    
    Args:
        frequencies (dict): Frequencies of outcomes.
        title (str): Title of the plot.
    """
    labels = list(measurement_dict.keys())
    values = list(measurement_dict.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Set x-axis ticks to integers only
    plt.xticks(ticks=range(len(labels)), labels=labels)
    
    plt.show()
    
    return plt













# Random state generation

def generate_haar_random_state(dimension: int):
    """
    Generate a Haar random quantum state in a specified dimension.
    """
    # Generate a random complex vector
    real_part = np.random.normal(size=dimension)
    imaginary_part = np.random.normal(size=dimension)
    random_complex_vector = real_part + 1j * imaginary_part

    # Compute the norm (magnitude) of the complex vector
    norm = np.linalg.norm(random_complex_vector)

    # Normalize the vector
    haar_random_state = random_complex_vector / norm     # No normalization check. I consider the system "if/not normalized" here.

    return {
        "Dimension": dimension,
        "Real Part": real_part.tolist(), 
        "Imaginary Part": imaginary_part.tolist(), 
        "Random Complex Vector": random_complex_vector.tolist(), 
        "Norm": norm, 
        "Normalized Haar Random State": haar_random_state.tolist(), 
        "Normalization Check": np.sum(np.abs(haar_random_state)**2)
    }


def generate_hilbert_schmidt_random_state(dimension: int):
    """
    Generate a Hilbert-Schmidt random quantum state (density matrix) in a given dimension.
    """
    # Step 1: Generate a random complex matrix G of size (dimension x dimension)
    real_part = np.random.normal(size=(dimension, dimension))
    imaginary_part = np.random.normal(size=(dimension, dimension))
    G = real_part + 1j * imaginary_part

    # Step 2: Compute GG^dagger
    GG_dagger = np.dot(G, G.conjugate().T)

    # Step 3: Normalize the matrix to make it a valid density matrix
    trace = np.trace(GG_dagger)  # Compute the trace of GG^dagger
    rho = GG_dagger / trace  # Normalize to ensure Tr(rho) = 1

    # Return the details
    return {
        "Dimension": dimension,
        "Random Complex Matrix G": G.tolist(),
        "GG^dagger": GG_dagger.tolist(),
        "Trace of GG^dagger": trace,
        "Normalized Density Matrix (rho)": rho.tolist(),
        "Normalization Check (Tr(rho))": np.trace(rho)
    }


def generate_bures_random_state(dimension: int):
    """
    Generate a Bures random quantum state (density matrix) in a given dimension.
    """
    # Step 1: Generate a random complex matrix G of size (dimension x dimension)
    real_part = np.random.normal(size=(dimension, dimension))
    imaginary_part = np.random.normal(size=(dimension, dimension))
    G = real_part + 1j * imaginary_part

    # Step 2: Compute GG^dagger
    GG_dagger = np.dot(G, G.conjugate().T)

    # Step 3: Add the identity matrix
    B = np.eye(dimension) + GG_dagger

    # Step 4: Normalize to create the density matrix
    trace_B = np.trace(B)
    rho = B / trace_B

    # Return the details
    return {
        "Dimension": dimension,
        "Random Complex Matrix G": G.tolist(),
        "GG^dagger": GG_dagger.tolist(),
        "Matrix B (I + GG^dagger)": B.tolist(),
        "Trace of B": trace_B,
        "Normalized Density Matrix (rho)": rho.tolist(),
        "Normalization Check (Tr(rho))": np.trace(rho)
    }












# channels

def bit_flip_channel(input_state, flip_probability, qubit = 1):
    """
    Args:
    - input_state: Input density matrix (2x2 numpy array).
    - flip_probability: Probability of a bit flip (0 ≤ p ≤ 1).
    """
    # create density matrix, if necessary
    if len(input_state.shape) == 1:
        input_state = input_state @ input_state.T
    else:
        if input_state.shape[0]>input_state.shape[1]:
            input_state = input_state @ input_state.T
    
    # Kraus operators
    no_flip = (1 - flip_probability) * input_state                
    
    instruction_list = ["X",[qubit]]
    instructions = create_instruction_list([instruction_list])
    flip = flip_probability * reduce(apply_instruction, instructions, input_state).T
    
    output_state = no_flip + flip
    
    return output_state


def phase_flip_channel(input_state, flip_probability, qubit = 1):
    """
    Args:
    - input_state: Input density matrix (2x2 numpy array).
    - flip_probability: Probability of a phase flip (0 ≤ p ≤ 1).
    """
    # create density matrix, if necessary
    if len(input_state.shape) == 1:
        input_state = input_state @ input_state.T
    else:
        if input_state.shape[0]>input_state.shape[1]:
            input_state = input_state @ input_state.T
    
    # Kraus operators
    no_flip = (1 - flip_probability) * input_state             
    
    instruction_list = ["Z",[qubit]]
    instructions = create_instruction_list([instruction_list])
    flip = flip_probability * reduce(apply_instruction, instructions, input_state).T        
    
    output_state = no_flip + flip
    
    return output_state


def amplitude_damping_channel(state, damping, qubit = 1):
    """    
    Args:
    - input_state: Input density matrix (2x2 numpy array).
    - damping: Probability of energy loss (0 ≤ gamma ≤ 1).
    - Excited to Ground state transition
    """
    # create density matrix, if necessary
    if len(state.shape) == 1:
        state = state @ state.T
    else:
        if state.shape[0]>state.shape[1]:
            state = state @ state.T
    
    qubit_amount = round(np.log2(len(state[0])))
    
    # Kraus operators for amplitude damping
    E0 = np.array([[1, 0], [0, np.sqrt(1 - damping)]])  # No decay
    E1 = np.array([[0, np.sqrt(damping)], [0, 0]])      # Decay from |1> to |0>
    
    E0 = single_qubit_gate_to_full_gate(E0, qubit_amount, qubit)
    E1 = single_qubit_gate_to_full_gate(E1, qubit_amount, qubit)
    
    # Conditions
    no_decay = E0 @ state @ E0.T.conj()  # E0 rho E0^†
    decay = E1 @ state @ E1.T.conj()     # E1 rho E1^†
    
    
    
    # Combine for results
    state = no_decay + decay
    return state


def depolarizing_channel(state, qubit_amount, damping):
    """    
    Args:
    - input_state: Input density matrix (2x2 numpy array).
    - qubit_amount: int
    - damping: Probability of depolarization.
    """
    state = damping * np.identity(2**qubit_amount,complex) / 2 + (1-damping) * state
    
    return state









# algorithms

def deutsch_josza(function, bits):
    """    
    Checks whether the function taking in a bitstring of length bits is 
    constant (1) or balanced (0)
    Args:
    - function: function in question {0,1}^n -> {0,1}.
    - bits: bitstring length the function takes
    Returns:
    - int (1 constant or 0 balanced)
    """
    # some helper function
    def indToState(n, k):
        num = bin(k)[2:].zfill(n)
        return np.array([int(x) for x in str(num)])
    

    
    # prepare state
    state = np.zeros((2**bits,))
    state[0] = 1
    
    # hadamard gates
    hadamard_instruction_list = ["H",[]]
    bit = 1
    while bit <= bits:
        hadamard_instruction_list[1].append(bit)
        bit += 1
    instructions = create_instruction_list([hadamard_instruction_list])
    state = reduce(apply_instruction, instructions, state).T
    
    # oracle
    element = 0
    while element < state.shape[0]:
        state[element] = state[element]*(-1)**function(indToState(bits, element))
        element += 1
    
    # hadamard round 2
    state = reduce(apply_instruction, instructions, state).T
    
    # measurement
    measurements = measure_computational(state,bits,1, False)
    
    key = ""
    bit = 1
    while bit <= bits:
        key += "0"
        bit += 1
    
    if(measurements[key]) == 1:
        return 1
    else:
        return 0
    

def deutsch_josza_noise(function, bits, noise_1 = None, p_1 = 0.0, noise_2 = None, p_2 = 0.0, noise_3 = None, p_3 = 0.0, noise_4 = None, p_4 = 0.0):
    """    
    Checks whether the function taking in a bitstring of length bits is 
    constant (1) or balanced (0)
    Args:
    - function: function in question {0,1}^n -> {0,1}.
    - bits: bitstring length the function takes
    Returns:
    - int (1 constant or 0 balanced)
    """
    # some helper function
    def indToState(n, k):
        num = bin(k)[2:].zfill(n)
        return np.array([int(x) for x in str(num)])
    

    
    # prepare state
    state = np.zeros((2**bits,))
    state[0] = 1
    state = np.outer(state, state.conj())
    
    # noise 1
    if noise_1:
        state = noise_1(state,p_1)
    
    # hadamard gates
    hadamard_instruction_list = ["H",[]]
    bit = 1
    while bit <= bits:
        hadamard_instruction_list[1].append(bit)
        bit += 1
    instructions = create_instruction_list([hadamard_instruction_list])
    state = reduce(apply_instruction, instructions, state).T
    
    # noise 2
    if noise_2:
        state = noise_2(state,bits,p_2)
    
    # oracle
    row = 0
    col = 0
    while row < state.shape[1]:
        col = 0
        while col < state.shape[0]:
            state[col,row] = state[col,row]*(-1)**function(indToState(bits, row))
            state[col,row] = state[col,row]*(-1)**function(indToState(bits, col))
            
            col += 1
        row += 1
    
    # noise 3
    if noise_3:
        state = noise_3(state,bits,p_3)
    
    # hadamard round 2
    state = reduce(apply_instruction, instructions, state).T
    
    # noise 4
    if noise_4:
        state = noise_4(state,bits,p_4)
    
    # measurement
    measurements = measure_computational(state,bits,1, False)
    
    key = ""
    bit = 1
    while bit <= bits:
        key += "0"
        bit += 1
    
    if(measurements[key]) == 1:
        return 1
    else:
        return 0