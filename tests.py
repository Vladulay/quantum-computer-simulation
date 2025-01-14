import unittest
import gate_operations as go
import random
import numpy as np
from functools import reduce
from scipy.sparse import csr_matrix

class TestGates(unittest.TestCase):

    def test_CNOT(self):
        qubit_amount = random.randrange(2, 6)
        control_qubit_index = random.randrange(1, qubit_amount)
        target_qubit_index = random.randrange(1, qubit_amount)
        
        CNOT1 = go.CNOT(qubit_amount, control_qubit_index, target_qubit_index)
        CNOT2 = go.CNOT_from_SWAP(qubit_amount, control_qubit_index, target_qubit_index)
        
        compare = np.allclose(CNOT1, CNOT2)
        
        self.assertTrue(compare)
    
    
    def test_CNOT_identity(self):
        qubit_amount = random.randrange(2, 6)
        control_qubit_index = random.randrange(1, qubit_amount)
        target_qubit_index = random.randrange(1, qubit_amount)
        
        CNOT = go.CNOT(qubit_amount, control_qubit_index, target_qubit_index)
        
        prod = np.matmul(CNOT, CNOT)
        
        ident = np.identity(2**qubit_amount)
                
        compare = np.allclose(prod, ident)
        
        self.assertTrue(compare)
    
    
    def test_H_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "H"
        instruction.qubit = [1]
        
        state1 = go.gate_operation(state, go.H())
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_double_H_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "H"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(go.H(), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(go.H(), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_X_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "X"
        instruction.qubit = [1]
        
        state1 = go.gate_operation(state, go.X())
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_double_X_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "X"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(go.X(), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(go.X(), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_Y_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "Y"
        instruction.qubit = [1]
        
        state1 = go.gate_operation(state, go.Y())
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_double_Y_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "Y"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(go.Y(), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(go.Y(), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)

    
    def test_Z_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "Z"
        instruction.qubit = [1]
        
        state1 = go.gate_operation(state, go.Z())
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_double_Z_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "Z"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(go.Z(), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(go.Z(), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_S_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "S"
        instruction.qubit = [1]
        
        state1 = go.gate_operation(state, go.S())
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_double_S_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "S"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(go.S(), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(go.S(), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_T_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "T"
        instruction.qubit = [1]
        
        state1 = go.gate_operation(state, go.T())
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_double_T_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "T"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(go.T(), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(go.T(), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_Rx_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        instruction.gate = "Rx"
        instruction.qubit = [1]
        instruction.angle = angle
        
        state1 = go.gate_operation(state, go.R_x(angle))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_double_Rx_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        instruction.gate = "Rx"
        instruction.qubit = [1,2]
        instruction.angle = angle
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(go.R_x(angle), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(go.R_x(angle), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_Ry_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        instruction.gate = "Ry"
        instruction.qubit = [1]
        instruction.angle = angle
        
        state1 = go.gate_operation(state, go.R_y(angle))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_double_Ry_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        instruction.gate = "Ry"
        instruction.qubit = [1,2]
        instruction.angle = angle
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(go.R_y(angle), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(go.R_y(angle), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_Rz_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        instruction.gate = "Ry"
        instruction.qubit = [1]
        instruction.angle = angle
        
        state1 = go.gate_operation(state, go.R_y(angle))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_double_Rz_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        instruction.gate = "Rz"
        instruction.qubit = [1,2]
        instruction.angle = angle
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(go.R_z(angle), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(go.R_z(angle), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_R_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        direction = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction.gate = "R"
        instruction.qubit = [1]
        instruction.angle = angle
        instruction.direction = direction
        
        state1 = go.gate_operation(state, go.R(direction, angle))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_double_R_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        direction = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction.gate = "R"
        instruction.qubit = [1,2]
        instruction.angle = angle
        instruction.direction = direction
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(go.R(direction, angle), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(go.R(direction, angle), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
        
    
    def test_R_instruction_no_input(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "R"
        instruction.qubit = [1]
        
        state1 = go.gate_operation(state, go.R(np.array([0,0,1]), 0.0))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_CNOT_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "CNOT"
        instruction.qubit = [2,1]
        
        state1 = go.gate_operation(state, go.CNOT(2, 2, 1))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_SWAP_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        instruction.gate = "SWAP"
        instruction.qubit = [2,1]
        
        state1 = go.gate_operation(state, go.SWAP(2, 2, 1))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state2)
        
        self.assertTrue(compare)
    
    
    def test_missing_instruction(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction = go.instruction()
        
        state1 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1, state)
        
        self.assertTrue(compare)
        
    
    def bell_states(self):
        H1 = go.single_qubit_gate_to_full_gate(go.H(), 2, 1)
        CNOT12 = go.CNOT(2,1,2)
        X1 = go.single_qubit_gate_to_full_gate(go.X(), 2, 1)
        X2 = go.single_qubit_gate_to_full_gate(go.X(), 2, 2)
        
        phi_plus  = np.array([1,0,0,0])
        phi_plus = go.gate_operation(phi_plus, H1)
        phi_plus = go.gate_operation(phi_plus, CNOT12)
        
        compare = np.allclose(phi_plus, np.array([0.70710678, 0, 0, 0.70710678]))
        
        self.assertTrue(compare)
        
        phi_minus  = np.array([1,0,0,0])
        phi_minus = go.gate_operation(phi_minus, X1)
        phi_minus = go.gate_operation(phi_minus, H1)
        phi_minus = go.gate_operation(phi_minus, CNOT12)
        
        compare = np.allclose(phi_minus, np.array([0.70710678, 0, 0, 0.70710678]))
        
        self.assertTrue(compare)
        
        psi_plus  = np.array([1,0,0,0])
        psi_plus = go.gate_operation(psi_plus, X2)
        psi_plus = go.gate_operation(psi_plus, H1)
        psi_plus = go.gate_operation(psi_plus, CNOT12)
        
        compare = np.allclose(psi_plus, np.array([0, 0.70710678, 0.70710678, 0]))
        
        self.assertTrue(compare)
        
        psi_minus  = np.array([1,0,0,0])
        psi_minus = go.gate_operation(psi_minus, X1)
        psi_minus = go.gate_operation(psi_minus, X2)
        psi_minus = go.gate_operation(psi_minus, H1)
        psi_minus = go.gate_operation(psi_minus, CNOT12)
        
        compare = np.allclose(psi_minus, np.array([0, 0.70710678, -0.70710678, 0]))
        
        self.assertTrue(compare)
    
    
    def GHZ_state(self):
        H1 = go.single_qubit_gate_to_full_gate(go.H(), 3, 1)
        CNOT12 = go.CNOT(3,1,2)
        CNOT23 = go.CNOT(3,2,3)
        
        GHZ = np.zeros((2**3,))
        GHZ[0] = 1
        
        GHZ = go.gate_operation(GHZ, H1)
        GHZ = go.gate_operation(GHZ, CNOT12)
        GHZ = go.gate_operation(GHZ, CNOT23)
        
        compare = np.allclose(GHZ, np.array([0.70710678, 0, 0, 0,0, 0, 0, 0.70710678]))
        
        self.assertTrue(compare)
    
    
    def compare_instructional_syntax(self):
        state = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        state = go.normalization_check(state)
        
        # Computation with direct gate functions
        gate1 = go.single_qubit_gate_to_full_gate(go.H(),3,1)
        gate2 = go.single_qubit_gate_to_full_gate(go.X(),3,2)
        gate3 = go.single_qubit_gate_to_full_gate(go.R_x(np.pi),3,2)
        gate4 = go.CNOT(3,2,1)
        gate5 = go.single_qubit_gate_to_full_gate(go.R(np.array([0,1,0]),np.pi),3,3)
        gate6 = go.single_qubit_gate_to_full_gate(go.T(),3,1)
        gate7 = go.single_qubit_gate_to_full_gate(go.T(),3,2)
        gate8 = go.single_qubit_gate_to_full_gate(go.T(),3,3)
        
        state1 = go.gate_operation(state, gate1)
        state1 = go.gate_operation(state1, gate2)
        state1 = go.gate_operation(state1, gate3)
        state1 = go.gate_operation(state1, gate4)
        state1 = go.gate_operation(state1, gate5)
        state1 = go.gate_operation(state1, gate6)
        state1 = go.gate_operation(state1, gate7)
        state1 = go.gate_operation(state1, gate8)
        
        
        # Computation with instruction list
        instructions = go.create_instruction_list([["H",[1]],
                                                   ["X",[2]],
                                                   ["Rx",[2],np.pi],
                                                   ["CNOT",[2,1]],
                                                   ["R",[3],np.pi,np.array([0,1,0])],
                                                   ["T",[1,2,3]]])
        
        state2 = reduce(go.apply_instruction, instructions, state)
        
        compare = np.allclose(state1,state2)
        
        self.assertTrue(compare)
    
    def test_csr_matrix_gate_operation(self):
        qubit_amount = random.randint(1, 6)

        # state prep |0000...0000〉
        state = np.zeros((2**qubit_amount,))
        state[0] = 1
        
        # non sparse
        Y = go.single_qubit_gate_to_full_gate(go.Y(), qubit_amount, random.randint(1, qubit_amount))
        
        # sparse
        Y_sparse = csr_matrix(Y)
        state_sparse = csr_matrix(state.T)

        sol1 = go.gate_operation(state, Y)

        sol2 = go.gate_operation(state_sparse, Y_sparse)
        
        #convert sparse back 
        sol2 = sol2.toarray().T
    
        
        compare = np.allclose(sol1,sol2)
        
        self.assertTrue(compare)


class TestCsrGates(unittest.TestCase):
    
    def test_CNOT(self):
        qubit_amount = random.randrange(2, 6)
        control_qubit_index = random.randrange(1, qubit_amount)
        target_qubit_index = random.randrange(1, qubit_amount)
        
        CNOT1 = go.CNOT_csr(qubit_amount, control_qubit_index, target_qubit_index).toarray()
        CNOT2 = go.CNOT(qubit_amount, control_qubit_index, target_qubit_index)
        
        compare = np.allclose(CNOT1, CNOT2)
        
        self.assertTrue(compare)
    
    
    def test_CNOT_identity(self):
        qubit_amount = random.randrange(2, 6)
        control_qubit_index = random.randrange(1, qubit_amount)
        target_qubit_index = random.randrange(1, qubit_amount)
        
        CNOT = go.CNOT_csr(qubit_amount, control_qubit_index, target_qubit_index)
        
        prod = (CNOT @ CNOT).toarray()
        
        ident = np.identity(2**qubit_amount)
        
        compare = np.allclose(prod, ident)
        
        self.assertTrue(compare)
    
    
    def test_H_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction("H",[1])
        
        state1 = go.gate_operation(state, csr_matrix(go.H()))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_double_H_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        instruction.gate = "H"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(csr_matrix(go.H()), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(csr_matrix(go.H()), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_X_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction("X",[1])
        
        state1 = go.gate_operation(state, csr_matrix(go.X()))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_double_X_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        instruction.gate = "X"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(csr_matrix(go.X()), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(csr_matrix(go.X()), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_Y_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction("Y",[1])
        
        state1 = go.gate_operation(state, csr_matrix(go.Y()))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_double_Y_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        instruction.gate = "Y"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(csr_matrix(go.Y()), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(csr_matrix(go.Y()), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)

    
    def test_Z_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction("Z",[1])
        
        state1 = go.gate_operation(state, csr_matrix(go.Z()))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_double_Z_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        instruction.gate = "Z"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(csr_matrix(go.Z()), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(csr_matrix(go.Z()), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_S_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction("S",[1])
        
        state1 = go.gate_operation(state, csr_matrix(go.S()))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_double_S_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        instruction.gate = "S"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(csr_matrix(go.S()), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(csr_matrix(go.S()), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_T_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction("T",[1])
        
        state1 = go.gate_operation(state, csr_matrix(go.T()))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_double_T_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        instruction.gate = "T"
        instruction.qubit = [1,2]
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(csr_matrix(go.T()), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(csr_matrix(go.T()), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_Rx_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1)]))
        angle = random.uniform(0,2 * np.pi)
        instruction = go.instruction("Rx",[1],angle)
        
        state1 = go.gate_operation(state, csr_matrix(go.R_x(angle)))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_double_Rx_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        instruction.gate = "Rx"
        instruction.qubit = [1,2]
        instruction.angle = angle
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(csr_matrix(go.R_x(angle)), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(csr_matrix(go.R_x(angle)), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_Ry_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1)]))
        angle = random.uniform(0,2 * np.pi)
        instruction = go.instruction("Ry",[1],angle)
        
        state1 = go.gate_operation(state, csr_matrix(go.R_y(angle)))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_double_Ry_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        instruction.gate = "Ry"
        instruction.qubit = [1,2]
        instruction.angle = angle
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(csr_matrix(go.R_y(angle)), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(csr_matrix(go.R_y(angle)), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_Rz_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1)]))
        angle = random.uniform(0,2 * np.pi)
        instruction = go.instruction("Rz",[1],angle)
        
        state1 = go.gate_operation(state, csr_matrix(go.R_z(angle)))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_double_Rz_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        instruction.gate = "Rz"
        instruction.qubit = [1,2]
        instruction.angle = angle
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(csr_matrix(go.R_z(angle)), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(csr_matrix(go.R_z(angle)), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_R_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        direction = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction.gate = "R"
        instruction.qubit = [1]
        instruction.angle = angle
        instruction.direction = direction
        
        state1 = go.gate_operation(state, csr_matrix(go.R(direction, angle)))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 1)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_double_R_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        angle = random.uniform(0,2 * np.pi)
        direction = np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])
        instruction.gate = "R"
        instruction.qubit = [1,2]
        instruction.angle = angle
        instruction.direction = direction
        
        state1 = go.gate_operation(state, go.single_qubit_gate_to_full_gate(csr_matrix(go.R(direction, angle)), 2, 1))
        state1 = go.gate_operation(state1, go.single_qubit_gate_to_full_gate(csr_matrix(go.R(direction, angle)), 2, 2))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_CNOT_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        instruction.gate = "CNOT"
        instruction.qubit = [2,1]
        
        state1 = go.gate_operation(state, go.CNOT_csr(2, 2, 1))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_SWAP_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        instruction.gate = "SWAP"
        instruction.qubit = [2,1]
        
        state1 = go.gate_operation(state, go.SWAP_csr(2, 2, 1))
        state2 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state2.toarray())
        
        self.assertTrue(compare)
    
    
    def test_missing_instruction(self):
        state = csr_matrix(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        instruction = go.instruction()
        
        state1 = go.apply_instruction(state, instruction, qubit_amount = 2)
        
        compare = np.allclose(state1.toarray(), state.toarray())
        
        self.assertTrue(compare)


class TestMeasurements(unittest.TestCase):

    def test_simple_states(self):
        state = np.array([1,0,0,0,0,0,0,0])
        
        result = go.measure_projective(state,3,100000,go.P_z(3))
        
        self.assertTrue(np.allclose(result[0], 1))
        
        result = go.measure_projective(state,3,100000,go.P_x(3))
        
        self.assertTrue(np.allclose(result[0], 1/(2**3),0.05))
        
        result = go.measure_projective(state,3,100000,go.P_y(3))
        
        self.assertTrue(np.allclose(result[0], 1/(2**3),0.05))
    
    
    def test_mixed_states(self):
        state = np.array([1,0,0,0,0,0,0,0])
        state = np.outer(state, state.conj())
        
        result = go.measure_projective(state,3,100000,go.P_z(3))
        
        self.assertTrue(np.allclose(result[0], 1))
        
        result = go.measure_projective(state,3,100000,go.P_x(3))
        
        self.assertTrue(np.allclose(result[0], 1/(2**3),0.05))
        
        result = go.measure_projective(state,3,100000,go.P_y(3))
        
        self.assertTrue(np.allclose(result[0], 1/(2**3),0.05))


class TestMixedStates(unittest.TestCase):

    def test_compare_evolution_via_gate_operation(self):
        pure_state = np.array([1,0,0,0,0,0,0,0])
        mixed_state = np.outer(pure_state, pure_state.conj())
        
        instructions = go.create_instruction_list([["H",[1]],
                                                   ["X",[2]]])
        
        pure_state =  reduce(go.apply_instruction, instructions, pure_state)
        
        mixed_state = go.gate_operation(mixed_state, go.single_qubit_gate_to_full_gate(go.H(), 3, 1), 3)
        mixed_state = go.gate_operation(mixed_state, go.single_qubit_gate_to_full_gate(go.X(), 3, 2), 3)
        
        new_mixed_state = np.outer(pure_state, pure_state.conj())
        
        compare = np.allclose(mixed_state, new_mixed_state)
                
        self.assertTrue(compare)
        
    def test_compare_evolution_via_instuction(self):
        pure_state = np.array([1,0,0,0,0,0,0,0])
        mixed_state = np.outer(pure_state, pure_state.conj())
        
        instructions = go.create_instruction_list([["H",[1]],
                                                   ["X",[2]],
                                                   ["Rx",[2],np.pi],
                                                   ["CNOT",[2,1]],
                                                   ["R",[3],np.pi,np.array([0,1,0])],
                                                   ["T",[1,2,3]]])
        
        pure_state =  reduce(go.apply_instruction, instructions, pure_state)
        
        mixed_state = reduce(go.apply_instruction, instructions, mixed_state)
        
        new_mixed_state = np.outer(pure_state, pure_state.conj())
        
        compare = np.allclose(mixed_state, new_mixed_state)
                
        self.assertTrue(compare)
    

class TestChannels(unittest.TestCase):

    def test_bitflip(self):
        state = np.array([1, 0],dtype = complex)
        state = np.outer(state, state.conj())
        
        state =  go.bit_flip_channel(state, 0.5,1)
        
        correct_state = np.array([[0.5, 0. ],[0. , 0.5]])
        
        compare = np.allclose(state, correct_state)
                
        self.assertTrue(compare)
    
    
    def test_bitflip2(self):
        state = np.array([[1, 0], [0, 0]],dtype = complex)
        
        instruction = go.instruction()
        instruction.gate = "bitflip"
        instruction.probability = 0.5
        
        state = go.apply_instruction(state, instruction,1)
        
        correct_state = np.array([[0.5, 0. ],[0. , 0.5]])
        
        compare = np.allclose(state, correct_state)
                
        self.assertTrue(compare)
        
        
    def test_bitflip3(self):
        state = np.array([[1, 0], [0, 0]])
        
        instructions = go.create_instruction_list([["bitflip",0.5,[1]]])
        
        state = reduce(go.apply_instruction, instructions, state)
        
        correct_state = np.array([[0.5, 0. ],[0. , 0.5]])
        
        compare = np.allclose(state, correct_state)
                
        self.assertTrue(compare)
    
    
    def test_phaseflip(self):
        state = np.array([1,0])
        state = go.gate_operation(state, go.R_y(np.pi/2))
        state = np.outer(state, state.conj())
        
        print(state)
        
        state2 = go.gate_operation(state, go.R_y(np.pi))

        instructions = go.create_instruction_list([["phaseflip",1,[1]]])
        
        state = reduce(go.apply_instruction, instructions, state)
        
        print(state)
        
        compare = np.allclose(state, state2)
                
        self.assertTrue(compare)
    
    
    def test_amplitude_damping(self):
        state  = np.array([[0, 0], [0, 1]])  # |1><1|

        instructions = go.create_instruction_list([["ampdamp",0.4,[1]]])
        
        state = reduce(go.apply_instruction, instructions, state)
        
        correct_state = np.array([[0.4, 0. ],[0.,  0.6]])
        
        compare = np.allclose(state, correct_state)
                
        self.assertTrue(compare)
        
    
    def test_depolarizing_channel(self):
        state  = np.array([[0, 0], [0, 1]])  # |1><1|

        instructions = go.create_instruction_list([["depol",0.5,1]])
        
        state = reduce(go.apply_instruction, instructions, state)
        
        correct_state = np.array([[0.25+0.j , 0.  +0.j],  [0.  +0.j, 0.75+0.j]])
        
        compare = np.allclose(state, correct_state)
                
        self.assertTrue(compare)
        

    
class TestAlgorithms(unittest.TestCase):

    def test_deutsch_josza(self):
        from deutsch_josza_black_box import black_box

        bits = 1
        
        while bits < 6:
            compare = np.allclose(go.deutsch_josza(black_box, bits), go.deutsch_josza_noise(black_box, bits))
            
            self.assertTrue(compare)
            
            bits += 1
    



if __name__ == '__main__':
    unittest.main()