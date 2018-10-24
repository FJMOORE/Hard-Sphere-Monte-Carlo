# -*- coding: utf-8 -*-
"""
Ideal gas: This script creates a lattice of non-interacting particles
to introduce the students to periodic boundary conditions.
"""
import random as random
import numpy as np

# Initial parameters
unit_repeat = 5  # number of unit cells per dimension
lattice_constant = 1
box_size = lattice_constant * unit_repeat
particle_number = unit_repeat ** 3  # number of particles
total_steps = 100
diameter = 1
volume_fraction = 0.1

# rewrite file
output_file = open('positions_random_gas.xyz', 'w')
output_file.close()

# Generate initial co-ordinates
x, y, z = [], [], []
for i in range(0, unit_repeat):
    for j in range(0, unit_repeat):
        for k in range(0, unit_repeat):
            x.append(i * lattice_constant)
            y.append(j * lattice_constant)
            z.append(k * lattice_constant)

# Rescale the box/coordinates so we achieve the correct volume fraction/density.
atoms_per_cell = 1
atom_volume = np.pi * diameter ** 3 / 6
initial_volume_fraction = atoms_per_cell * atom_volume  # / unit volume (=1)
rescale = (initial_volume_fraction / volume_fraction) ** (1. / 3)
box_size *= rescale

for a in range(0, particle_number):
    x[a] *= rescale
for a in range(0, particle_number):
    y[a] *= rescale
for a in range(0, particle_number):
    z[a] *= rescale

# Move particles, output their coordinates
for t in range(0, total_steps):
    # Write positions to file
    with open('positions_random_gas.xyz', 'a') as f:
        f.write(str(len(x)) + '\n')
        f.write('\n')
        for i in range(len(x)):
            # xyz file is a file format to store 3D position
            # the general format is:
            # PARTICLE_TYPE  X  Y  Z
            # here we just call our random gas particles "G"
            f.write('G' + '\t' + str(x[i]) + '\t' + str(y[i]) + '\t' + str(z[i]) + '\n')

    for i in range(0, particle_number):
        # Trial Move
        trial_x = x[i] + random.gauss(0, 1)
        trial_y = y[i] + random.gauss(0, 1)
        trial_z = z[i] + random.gauss(0, 1)

        # Check boundaries
        # We always move particles a small step, so don't worry if trial_x >> box_size
        if trial_x <= 0:
            trial_x += box_size
        elif trial_x >= box_size:
            trial_x -= box_size

        if trial_y <= 0:
            trial_y += box_size
        elif trial_y >= box_size:
            trial_y -= box_size

        if trial_z <= 0:
            trial_z += box_size
        elif trial_z >= box_size:
            trial_z -= box_size

        # Confirm movements
        x[i], y[i], z[i] = trial_x, trial_y, trial_z
