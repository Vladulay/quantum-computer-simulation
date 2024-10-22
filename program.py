# Hi
import gate_operations as go
import numpy as np
import qutip as qt


# INPUT SPACE
in_state = np.array([5,3j+5])

gate = np.array([[0,-1j],[1j,0]])

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
