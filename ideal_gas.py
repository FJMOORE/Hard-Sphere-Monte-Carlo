"""
Ideal gas: This script creates a lattice of non-interacting particles 
to introduce the students to periodic boundary conditions.
"""
import random as random
import numpy as np

#Initial parameters
L = 5 #lattice unit cells per dimention
a = 1 #lattice parameter
box = L * a
N = (L)**3 #number of particles
Total_steps = 100
diameter = 1
volume_fraction = 0.1


#rewrite file
myfile = open('positions.txt' , 'w')
myfile.close()

#Generate initial co-ordinates
x=[]
y=[]
z=[]
for i in range(0,L):
    for j in range(0,L):
        for k in range(0,L):
            x.append(i*a)
            y.append(j*a)
            z.append(k*a) 

            
# Rescale the box/coordinates so we achieve the correct volume fraction/density.
atoms_per_cell = 1
atom_volume = np.pi*diameter**3 / 6
initial_volume_fraction = atoms_per_cell * atom_volume # / unit volume (=1)
rescale = (initial_volume_fraction/volume_fraction)**(1./3)
box *= rescale 
for a in range(0,N):
    x[a] *= rescale
for a in range(0,N):
    y[a] *= rescale
for a in range(0,N):
    z[a] *= rescale

#Move particles        
for t in range(0,Total_steps):   
    
    #Write positions to file
    with open('positions.txt','a') as f:
        f.write(str(len(x)) + '\n')
        f.write('\n')
        for i in range(len(x)):
            f.write('1' + '\t' + str(x[i]) + '\t' + str(y[i]) + '\t' + str(z[i]) + '\n')
                        
    for i in range(0,N):        
        #Trial Move
        trial_x = x[i] + random.gauss(0,1)
        trial_y = y[i] + random.gauss(0,1)
        trial_z = z[i] + random.gauss(0,1)
        
        #Check boundaries       
        if trial_x <= 0:
            trial_x += box
        elif trial_x >= box:
            trial_x -= box    
        
        if trial_y <= 0:
            trial_y += box
        elif trial_y >= box:
            trial_y -= box 
            
        if trial_z <= 0:
            trial_z += box
        elif trial_z >= box:
            trial_z -= box

        #Confirm movements
        x[i],y[i],z[i] = trial_x,trial_y,trial_z       
        
