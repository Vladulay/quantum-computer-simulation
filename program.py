# Hi
import gate_operations as go
import numpy as np
import qutip as qt

# GOALS
# - Extend and play around more with the Bloch sphere visualizer by implementing all the standard gates from NS.
# - Implement the gate from eq. (4.8) in NS, convince yourself that it implements the rotation about the axis defined by n.
# - Create a framework for single qubit circuit creation.
# - Extend the circuit framework to 2 qubits, where we can apply any one- and two-qubit gates between the qubits.
# - Look into/implement some automated testing of your functions inside python (see e.g. Unittest). 










# INPUT SPACE
in_state = np.array([1,1])

gate = go.T()

assure_normalization = True
print_bloch_components = True


















# create plot
bloch = qt.Bloch()

if assure_normalization:
    in_state = go.normalization_check(in_state)

# plot of in state
bloch.add_vectors(go.spherical_to_cartesian(go.transform_to_bloch_vector(in_state)))

out_state = go.single_qubit_gate_operation(in_state, gate)

bloch.add_vectors(go.spherical_to_cartesian(go.transform_to_bloch_vector(out_state)))

bloch.show()

if print_bloch_components:
    print("In: " + str(go.transform_to_bloch_vector(in_state)))
    print("Out: " + str(go.transform_to_bloch_vector(out_state)))
