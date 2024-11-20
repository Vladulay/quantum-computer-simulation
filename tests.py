import unittest
import gate_operations as go
import random
import numpy as np
from functools import reduce
from scipy.sparse import lil_array
from scipy.sparse import csr_matrix
import scipy.sparse

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
    
    
    def test_double_T_instruction(self):
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


    


if __name__ == '__main__':
    unittest.main()