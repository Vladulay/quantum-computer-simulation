import unittest
import gate_operations as go
import random
import numpy as np

# EXAMPLE CODE
class TestMethods(unittest.TestCase):

    def test_CNOT(self):
        qubit_amount = random.randrange(2, 6)
        control_qubit_index = random.randrange(1, qubit_amount)
        target_qubit_index = random.randrange(1, qubit_amount)
        
        CNOT1 = go.CNOT(qubit_amount, control_qubit_index, target_qubit_index)
        CNOT2 = go.CNOT_from_SWAP(qubit_amount, control_qubit_index, target_qubit_index)
        
        compare = np.isclose(CNOT1, CNOT2)
        result = bool()
        
        if False in compare:
            result = False
        else: 
            result = True
        
        self.assertTrue(result)



if __name__ == '__main__':
    unittest.main()