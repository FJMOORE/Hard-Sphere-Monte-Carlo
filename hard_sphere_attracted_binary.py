# -*- coding: utf-8 -*-
"""
This script creates a lattice of two types of particles that won't overlap and attracting each other
The attraction is a square well potential, different particles have different interactions.
You can easily see the attraction from the g(r)
The python version of the code may not be fast enough to see the equilibriumed structure
Please contack Yushi (yushi.yang@bristol.ac.uk) for a faster C++ version (written by Paddy Royall)
(If you are interested in the system)
"""
import random as random
import numpy as np


def get_distance_in_pbc(p1, p2, box):
    dimension = len(p1)
    distance_nd = []
    for d in range(dimension):
        distance_1d = abs(p2[d] - p1[d])
        if distance_1d > (box[d] / 2):
            distance_nd.append((box[d] - distance_1d) ** 2)
        else:
            distance_nd.append(distance_1d ** 2)
    distance = sum(distance_nd) ** 0.5
    return distance

def check_overlap(p1, p2, diameter_1, diameter_2, box):
    distance = get_distance_in_pbc(p1, p2, box)
    if distance > (diameter_1 + diameter_2)/2:
        return False  # not overlap
    else:
        return True

def get_energy(i, system, depth_table, width_table, diameters, box):
    energy = 0
    p1 = system[i]
    for j, p2 in enumerate(system):  # j: indices of corresponding particels in the system
        if i != j:
            distance = get_distance_in_pbc(p1, p2, box)
            interaction_range = width_table[i][j] + (diameters[i] + diameters[j]) / 2
            if distance <= interaction_range:
                energy = energy + depth_table[i][j]
    return energy

def get_system_energy(system, depth_table, width_table, diameters, box):
    """calculate the energy for all particles"""
    energy = 0
    for i, p1 in enumerate(system):
        for j, p2 in enumerate(system[i + 1:]):
            j = j + i + 1
            distance = get_distance_in_pbc(p1, p2, box)
            if distance <= width_table[i][j] + (diameters[i] + diameters[j]) / 2:
                energy = energy + depth_table[i][j]
    return energy

def adjust_speed(old_speed, accept_ratio, box_size):
    if accept_ratio > 0.5:
        new_speed = 1.1 * old_speed
    elif accept_ratio < 0.45:
        new_speed = 0.9 * old_speed
    return max([new_speed, box_size/50.0])


# parameters about the box
unit_repeat = 5  # number of unit cells per dimension
lattice_constant = 1
box_size = lattice_constant * unit_repeat
particle_number = unit_repeat ** 3  # number of particles
volume_fraction = 0.1
ratio_ab = 0.5   # larger ratio --> more type A particle

# parameters about two types of particles
diameter_a = 1
diameter_b = 1

# parameters about the square well
depth_aa = -5
depth_bb = -5
depth_ab = 5
width_aa = 0.1 * diameter_a
width_bb = 0.1 * diameter_b
width_ab = 0.5 * (diameter_a + diameter_b) / 2

# parameters about the simulation
before_equilibrium = 0   # the frames before this number won't be outputed
total_steps = 500

# rewrite file
output_file = open('positions_hard_sphere_attracted_binary.xyz', 'w')
output_file.close()

# since the particles are different now, we need to know which position belongs to which type of particles
# we call it "labels", since every particles have different labels, A or B
labels = []
# also, let's construct a diameter list so to use later
diameters = []
for i in range(particle_number):
    flipped_coin = random.random()
    if flipped_coin <= ratio_ab:
        labels.append('A')
        diameters.append(diameter_a)
    else:
        labels.append('B')
        diameters.append(diameter_b)


# Let's construct some tables for the interaction
depth_table, width_table = [], []
for i in range(particle_number):
    depth_table.append([])
    width_table.append([])
    for j in range(particle_number):
        if i == j:
            depth_table[i].append(None)
            width_table[i].append(None)
        else:
            if labels[i] != labels[j]:
                depth_table[i].append(depth_ab)
                width_table[i].append(width_ab)
            elif labels[i] == 'A':
                depth_table[i].append(depth_aa)
                width_table[i].append(width_aa)
            else:
                depth_table[i].append(depth_bb)
                width_table[i].append(width_bb)

# Generate initial co-ordinates
x, y, z = [], [], []
for i in range(0, unit_repeat):
    for j in range(0, unit_repeat):
        for k in range(0, unit_repeat):
            x.append(i * lattice_constant)
            y.append(j * lattice_constant)
            z.append(k * lattice_constant)


# Rescale the box/coordinates for correct volume fraction/density.
number_a = sum([1 for label in labels if label == 'A'])
number_b = particle_number - number_a
volume_a = np.pi * diameter_a ** 3 / 6 * number_a
volume_b = np.pi * diameter_b ** 3 / 6 * number_b
initial_volume_fraction =  (volume_a + volume_b) / box_size ** 3
rescale = (initial_volume_fraction / volume_fraction) ** (1. / 3)
box_size *= rescale
box = [box_size, box_size, box_size]


for a in range(0, particle_number):
    x[a] *= rescale
for a in range(0, particle_number):
    y[a] *= rescale
for a in range(0, particle_number):
    z[a] *= rescale

# Move particles, output their coordinates
for t in range(0, total_steps + before_equilibrium):
    accept_count = 0
    # Write positions to file
    if t > before_equilibrium:
        with open('positions_hard_sphere_attracted_binary.xyz', 'a') as f:
            f.write(str(len(x)) + '\n')
            f.write('box is {}, at frame {}\n'.format(box, t))
            for i in range(len(x)):
                # xyz file is a file format to store 3D position
                # the general format is:
                # PARTICLE_TYPE  X  Y  Z
                f.write(labels[i] + '\t' + str(x[i]) + '\t' + str(y[i]) + '\t' + str(z[i]) + '\n')


    if t == 0:
        speed = box_size / 10  # same value as paddy
        print(speed)
    else:
        speed = adjust_speed(speed, accept_count / particle_number, box_size)
        #print('new speed is ', speed)

    for i in range(0, particle_number):
        i = random.randint(0, particle_number - 1)  # randomly pick up a particle and move
        trial_x = x[i] + random.uniform(-1, 1) * speed
        trial_y = y[i] + random.uniform(-1, 1) * speed
        trial_z = z[i] + random.uniform(-1, 1) * speed

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

        p1 = [trial_x, trial_y, trial_z]
        # check if the trial particle is overlaping with other particles
        is_overlap = False
        for j in range(0, particle_number):
            if is_overlap:
                break
            if i != j:
                p2 = [x[j], y[j], z[j]]
                is_overlap = check_overlap(p1, p2, diameters[i], diameters[j], box)

        if not is_overlap:
            # Confirm movements
            old_system = np.vstack([x, y, z]).T  # ((x1, x2, ..), (y1, y2, ..), (z1, z2, ..)) --> ((x1, y1, z1), (x2, y2, z2) ...)

            new_system = np.delete(old_system, i, axis=0)  # old one won't change
            new_system = np.insert(new_system, i, np.array(p1), axis=0)

            old_energy = get_energy(i, old_system, depth_table, width_table, diameters, box)
            new_energy = get_energy(i, new_system, depth_table, width_table, diameters, box)

            delta = new_energy - old_energy
            accept_probability = np.exp(-1 * delta)

            # if probability is HIGH, a random number is less likely to be higher than it
            if random.random() < accept_probability:  
                accept_count += 1
                x[i], y[i], z[i] = trial_x, trial_y, trial_z

    new_system = np.vstack([x, y, z]).T  # ((x1, x2, ..), (y1, y2, ..), (z1, z2, ..)) --> ((x1, y1, z1), (x2, y2, z2) ...)
    print('energe per atom is ', get_system_energy(new_system, depth_table, width_table, diameters, box) / particle_number)
