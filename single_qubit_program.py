import numpy as np
import gate_operations as go
import qutip as qt

# Only visualize seperate states by calls (custom colors etc)

# INPUT SPACE
state = np.array([1,0])

assure_normalization = True








# create plot
bloch_sphere = qt.Bloch()



if assure_normalization:
    state = go.normalization_check(state)






# SEQUENCE
state = go.gate_operation(state, go.H())
state = go.gate_operation(state, go.R_z(np.pi/2))
state = go.gate_operation(state, go.R_x(np.pi))
state = go.gate_operation(state, go.R_x(np.pi/2))
state = go.gate_operation(state, go.R_y(np.pi/2))

go.plot_bloch_state(state, bloch_sphere, "green")
















bloch_sphere.show()