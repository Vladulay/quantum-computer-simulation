import unittest
import gate_operations as go
import random
import numpy as np

class TestGates(unittest.TestCase):

    def test_CNOT(self):
        qubit_amount = random.randrange(2, 6)
        control_qubit_index = random.randrange(1, qubit_amount)
        target_qubit_index = random.randrange(1, qubit_amount)
        
        CNOT1 = go.CNOT(qubit_amount, control_qubit_index, target_qubit_index)
        CNOT2 = go.CNOT_from_SWAP(qubit_amount, control_qubit_index, target_qubit_index)
        
        compare = np.allclose(CNOT1, CNOT2)
        
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
        



if __name__ == '__main__':
    unittest.main()