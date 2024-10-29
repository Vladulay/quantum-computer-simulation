import numpy as np
import gate_operations as go
import qutip as qt

# INPUT SPACE
state = np.array([1,0])

assure_normalization = True








# create plot
bloch = qt.Bloch()



if assure_normalization:
    state = go.normalization_check(state)






# SEQUENCE
state = go.plotted_single_qubit_operation(state, go.H(), bloch)
state = go.plotted_single_qubit_operation(state, go.R_z(np.pi/2), bloch)

















bloch.show()