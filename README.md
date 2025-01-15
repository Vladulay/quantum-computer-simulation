# Quantum Computing Simulation

Here we develop a framework for quantum simulation. Quickly apply quantum circuits to pure & mixed states and perform measurements. Plot single qubits on the bloch sphere. Apply noise via channels and test its influence on the Deutsch-Josza algorithm.

## Features
- Gate & channel application via instruction list syntax
- Measurements in arbitary bases via projectors
- Supports both pure and mixed states
- Supports sparse matrices for pure states
- Deutsch-Josza algorithm

## How to use
The functions of interest are located inside _gate_operations.py_. To see examples of how to use this library consult the jupyter notebook _tutorial.ipynb_. The jupyter notebook _noisy deutsch josza.ipynb_ includes a script to test the influence of different noise types on the Deutsch-Josza algorithm.. 

### Single Qubit State Visualizer
To visualize single qubit states between any transformations one ca use the function _plot_bloch_state_.

### Gate Operations
There are several ways to go about applying gate operations. 
- One can directly call the function _gate_operation_ and specify the state and gate, to return a transformed state.
- One can use the _apply_instruction_ function. For this, one has to create an instance of the _instruction_ class and define gate and target qubits, which are properties of this object.
- One can apply a list of instructions with the _reduce_ function from the _functools_ package and _apply_instruction_. This is the simplest way to quickly apply large circuits. 

### Instruction Class
Has four properties, which one can define in a list in this order to use the instructional list syntax
 - _gate_ (String) which specifies the gate one wants to apply
 - _qubit_ (List) which qubit indices to apply the gates to (beginning with 1)
 - _angle_ (float) how much to rotate (if rotation gate)
 - _direction_ (np.array) what axis to rotate about (if general rotation gate)

The following instructions are possible:
- ["H",[1,2,5]] - apply Hadamard gate to qubit 1,2 and 5
- ["X",[1,2,5]] - apply Pauli X gate to qubit 1,2 and 5
- ["Y",[1,2,5]] - apply Pauli X gate to qubit 1,2 and 5
- ["Z",[1,2,5]] - apply Pauli X gate to qubit 1,2 and 5
- ["S",[1,2,5]] - apply phase gate to qubit 1,2 and 5
- ["T",[1,2,5]] - apply $\pi/8$ gate to qubit 1,2 and 5
- ["Rx",[1,2,5], np.pi] - apply x-rotation gate to qubit 1,2 and 5 with the angle $\pi$
- ["Ry",[1,2,5], np.pi] - apply y-rotation gate to qubit 1,2 and 5 with the angle $\pi$
- ["Rz",[1,2,5], np.pi] - apply z-rotation gate to qubit 1,2 and 5 with the angle $\pi$
- ["Rx",[1,2,5], np.pi, np.array([0,1,0])] - apply x-rotation gate to qubit 1,2 and 5 with the angle $\pi$ about the axis [0,1,0]
- ["CNOT",[2,1]] - apply CNOT gate with the control qubit 2 and the target qubit 1
- ["SWAP",[2,1]] - apply SWAP gate to the qubits 2 and 1
- ["bitflip",0.3,[1]] - apply bitflip channel with flip probability 0.3 to qubit 1
- ["phaseflip",0.3,[1]] - apply phaseflip channel to SINGLE QUBIT with flip probability 0.3 to qubit 1
- ["ampdamp",0.3,[1]] - apply amplitude damping channel to SINGLE QUBIT with damping probability 0.3 to qubit 1
- ["depol",0.3] - apply depolarization channel with depolarization probability 0.3
  

### Measurements
A state can be measured a given amount of times via the _measure_projective_ function in any arbitrary projector basis. The standard bases can be produced by the functions _P_x_, _P_y_ and _P_z_. The computational basis can be used directly via the _measure_computational_ function.

## Change Log
08.01.2025
+ added some quantum channels
+ added the Deutsch-Josza algorithm

18.12.2024
+ added full mixed state support (except for sparse matrices)
+ added random state generation functions

08.12.2024
+ Added measurement framework
+ **Finished basic pure state framework**
+ added mixed state bloch plotting

27.11.2024
+ Added scipy sparse compatability for instructional syntax
+ Code speedup for dense matrices through internal use of sparse matrices
+ Added benchmark data inside notebook 
+ Some measurement experimentation in the notebook

20.11.2024
+ Improved instruction class
+ Added perfomance tests for numpy arrays to jupyter notebook
+ Added scipy csr matrix compatability to gate_operation function (more to come)

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
