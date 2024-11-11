# Arbitrary CNOT constructed from SWAPS

import gate_operations as go
import numpy as np

qubit_amount = 7
control_index = 5
target_index = 3


print("Correct CNOT:")
CNOT = go.CNOT(qubit_amount,control_index,target_index)
print(CNOT)
print("CNOT constructed from CNOT with control 1 and target 2 and SWAPs:")

CNOT_constructed = go.SWAP(qubit_amount, 1, control_index)      # SWAP controls & targets to 1 & 2  
if target_index != 1:      
    CNOT_constructed = np.matmul(go.SWAP(qubit_amount, 2, target_index) ,CNOT_constructed)
    
    
    CNOT_constructed = np.matmul(go.CNOT(qubit_amount,1,2) ,CNOT_constructed) # apply CNOT
    
    
    CNOT_constructed = np.matmul(go.SWAP(qubit_amount, 2, target_index) ,CNOT_constructed)
    CNOT_constructed = np.matmul(go.SWAP(qubit_amount, 1, control_index) ,CNOT_constructed)  # SWAP back
else: 
    if control_index != 2:
        CNOT_constructed = np.matmul(go.SWAP(qubit_amount, 2, control_index) ,CNOT_constructed)
    
    CNOT_constructed = np.matmul(go.CNOT(qubit_amount,1,2) ,CNOT_constructed) # apply CNOT
    
    if control_index != 2:
        CNOT_constructed = np.matmul(go.SWAP(qubit_amount, 2, control_index) ,CNOT_constructed)
    
    CNOT_constructed = np.matmul(go.SWAP(qubit_amount, 1, control_index) ,CNOT_constructed) # SWAP back


print(CNOT_constructed)

#CNOT_constructed = go.CNOT_from_SWAP(qubit_amount, control_index, target_index)

print("Are they the same?")
compare = np.isclose(CNOT_constructed, CNOT)
if False in compare:
    print("No")
else: 
    print("Yes")