### quantum-computer-simulation

Here we develope a framework for quantum simulation


### How to use
The functions of interest are located inside "gate_operations.py". The two example scripts are "single_qubit_program.py" and "multi_qubit_program.py". 

## single_qubit_program.py
Define an initial state and a sequence of plotted gate operations to directly see the gate effects on the bloch sphere

## multi_qubit_program.py
Define a qubit amount and a gate sequence. The function "single_qubit_gate_to_full_gate" tensors any single qubit gate into the appropriate multi qubit matrix form.

### Change Log
29.10.2024
+ Added multi qubit framework (CNOT is still limited)
+ Added standard gates for quick reference

24.10.2024
+ Added simple one qubit + gate framework with visualizations on the bloch sphere
