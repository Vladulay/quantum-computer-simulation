# quantum-computer-simulation

Here we develope a framework for quantum simulation


## How to use
The functions of interest are located inside _gate_operations.py_. To see examples of how to use this library consult the jupyter notebook _tutorial.ipynb_. 

### Single Qubit State Vsualizer
To visualize single qubit states between any transformations one ca use the function _plot_bloch_state_.

### Gate Operations
There are several ways to go about applying gate operations. 
- One can directly call the function _gate_operation_ and specify the state and gate, to return a transformed state.
- One can use the _apply_instruction_ function. For this, one has to create an instance of the _instruction_ class and define gate and target qubits, which are properties of this object.
- One can apply a list of instructions with the _reduce_ function from the _functools_ package and _apply_instruction_list_. This is the simplest way to quickly apply large circuits.

## Change Log
12.11.2024
+ Added tutorial jupyter notebook
+ Added instructional syntax
+ Added more unittests

06.11.2024
+ Added seperate state plot function for Bloch visualizer with color specfication
+ Implemented a SWAP gate between arbitrary qubits
+ Implemented a CNOT gate between arbitrary qubits
+ Added unittest script

29.10.2024
+ Added multi qubit framework (CNOT is still limited to always use qubit 1 as control and qubit 2 as target)
+ Added standard gates for quick reference

24.10.2024
+ Added simple one qubit + gate framework with visualizations on the bloch sphere
