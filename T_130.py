# %%

import math
import random
import numpy as np 
import matplotlib.pyplot as plt
import copy
import sys
from contextlib import redirect_stdout
from numba import jit, cuda
import multiprocessing

# Gibbs Ensemble Monte Carlo code 
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Arrange atoms in a Lattice
def Lattice(Npart, Length):
    #Npart = int(input("Enter Npart:"))
    #Box = int(input("Enter box length:"))
    
    rx = []
    ry = []
    rz = []
    K = []

    N = int((Npart)**(1/3))+1
    if N == 0:
        N = 1     
    Del = Length/float(N)
    Itel = 0
    Dx = -Del
    #print(Itel);
    for I in range(0, N, 1):
        Dx = Dx + Del
        Dy = -Del
        for J in range(0, N, 1):
            Dy = Dy + Del
            Dz = -Del
            for K in range(0, N, 1):
                Dz = Dz + Del
                if Itel < Npart:
                    Itel = Itel + 1
                    #print(Itel)
                    rx.append(Dx)
                    ry.append(Dy)
                    rz.append(Dz)  
    #K= X
    #print(K)  
    return rx, ry, rz

def plot3(Box):
    k = np.zeros((N,3))
    for i in range(0, int(n[0])):
        k[i, :] = r[:, i]
    ax = plt.axes(projection = '3d')
    zdata = k[:,2]
    xdata = k[:,0]
    ydata = k[:,1]
    print('Box 1')
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
    plt.show()
    k = np.zeros((N,3))
    for i in range(int(n[0]), int(n[0]+n[1])):
        k[i, :] = r[:, i]
    ax = plt.axes(projection = '3d')
    zdata = k[:,2]
    xdata = k[:,0]
    ydata = k[:,1]
    print('Box 2')
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
    plt.show()
    return

def calc_variables(box, n, m1_ratio, m2_ratio, x12_ratio, x21_ratio, v_ratio, temperature, total, r_cut):
    #variables = np.zeros(13)
    vol = np.array([0, 0])
    #rho = np.array([0, 0])
    
    vol[:] = box[:]**3
    density_1 = n[0]/vol[0]
    density_2 = n[1]/vol[1]
    
    m1_r = m1_ratio
    m2_r = m2_ratio
    x12_r = x12_ratio
    x21_r = x21_ratio
    v_r = v_ratio
    
    n_1 = n[0]
    n_2 = n[1]
    
    #density_1 = rho[0]
    #density_2 = rho[1]
    
    e1_c = (1.5*temperature) + (total[0,0]/n[0])
    e2_c = (1.5*temperature) + (total[1,0]/n[1])
    
    p1_c = pressure_delta(density_1, r_cut) + (density_1*temperature) + (total[0,1]/vol[0])
    p2_c = pressure_delta(density_2, r_cut) + (density_2*temperature) + (total[1,1]/vol[1])
    
    variables = np.array([m1_r, m2_r, x12_r, x21_r, v_r, n_1, n_2, density_1, density_2, e1_c, e2_c, p1_c, p2_c])
    
    return variables

def move(i, ri, r):
    r[:, i] = ri
    return 

def swap(i, ri, r, n):
    if (i <= n[0]-1):
        t = int(n[0] - 1)                  # Last atom in system 1
        r[:, i] = r[:, t]             # Replace i coord. with t coord.
        r[:, t] = ri
        n[:] = n[:] + [-1, 1]        # Move Boundary up to include t
    
    else:
        t = int(n[0])
        r[:, i] = r[:, t]
        r[:, t] = ri
        n[:] = n[:] + [1, -1]
        
    if (np.min(n) <= 0):
        sys.exit('Number of particles is zeros in a box')
    return  

def introduction():
    print('LJ Potential:  Diameter = 1,  Sigma = 1', file = blk_file)
    print('Epsilon = 1, Two Simulation Boxes', file = blk_file)
    return

def conclusion():
    print("-------------   Program Ends   -------------------", file = blk_file)
    return

def potential_1(i1, i2, ri, i, boxn, r_cut):
    rij = np.zeros(3)
    pair = np.array([0, 0, False])
    sr2_ovr = 1.77
    
    j1 = i1
    j2 = i2
    
    r_cut_box = r_cut / boxn
    r_cut_box_sq = r_cut_box**2
    box_sq = boxn**2
    
    partial = np.array([0, 0 ,False])
    
    for j in range(int(j1), int(j2)):
        if i==j: continue
        rij[:] = ri[:] - r[:, j]
        rij = pbc_1(rij)
        rij_sq = np.sum(rij**2)
        
        if rij_sq < r_cut_box_sq : 
            rij_sq = rij_sq*box_sq
            sr2 = 1.0/rij_sq
            pair[2] = (sr2 > sr2_ovr)
            #if (sr2 > sr2_ovr) == True: pair[2] = True
            
            if pair[2] == True: 
                partial[2] = True
                return partial
            
            sr6 = sr2**3
            sr12 = sr6**2
            pair[0] = sr12 - sr6
            pair[1] = pair[0] + sr12
            
            partial = partial + pair
            
    partial[0] = partial[0] * 4.0
    partial[1] = partial[1] * (24.0/3.0)            
    partial[2] = False
    
    return partial

def potential(i1, i2, boxn, r_cut):
    total_p = np.array([0, 0, False])
    partial = np.array([0, 0, False])
    box_n = boxn
    for i in range(int(i1), int(i2)):
        partial = potential_1(i1, i2, r[:,i], i, box_n, r_cut)
        #if partial[2] == True: 
        #    total_p[2] = True
        #    return
        
        total_p = total_p + partial
    
    total_p[2] = False
    return total_p

def run_begin():
    run_avg = np.zeros(13)
    run_err = np.zeros(13)
    run_nrm = 0
    return run_avg, run_err, run_nrm
    
def blk_begin():
    blk_avg = np.zeros(13)
    blk_msd = np.zeros(13)
    blk_nrm = 0
    return blk_avg, blk_msd, blk_nrm

def blk_add(variables, blk_avg, blk_msd, blk_nrm):
    blk_avg = blk_avg + variables
    blk_msd = blk_msd + (variables**2)
    blk_nrm = blk_nrm + 1.0
    return blk_avg, blk_msd, blk_nrm

def blk_end(blk_avg, blk_msd, blk_nrm, run_avg, run_err, run_nrm):
    if blk_nrm < 0.5: sys.exit('Block Accumulation Error')
    blk_avg = blk_avg/blk_nrm
    blk_msd = blk_msd/blk_nrm
    
    run_avg = run_avg + blk_avg
    run_err = run_err + (blk_avg**2)
    run_nrm = run_nrm + 1.0
    
    #print('Final Average Calculations at the end of block:   ', blk_avg)
    return blk_avg, blk_msd, blk_nrm, run_avg, run_err, run_nrm

def run_end(variables, run_avg, run_err, run_nrm):
    run_avg = run_avg / run_nrm    # Normalize run averages
    run_err = run_err / run_nrm    # Normalize error estimates
    run_err = run_err - run_avg**2 # Compute fluctuations of block averages
    
    for i in range(0, 13):
        if run_err[i] > 0.0:
            run_err[i] = np.sqrt(run_err[i]/run_nrm)
        
    #print('Final Average Calculations at the end of block:   ')
    #print(vairables)
    return run_avg, run_err, run_nrm

def random_translate_vector(dr_max, old):
    zeta = np.zeros(3)
    rr=np.zeros(3)
    zeta = np.random.random(3)
    zeta = 2*zeta - 1
    rr[:] = old[:] + (zeta*dr_max)
    return rr

def metropolis(delta):
    exponent_guard = 75.0
    if (delta > exponent_guard): accept = False
    elif (delta < 0): accept = True
    else:
        zeta = random.random()
        if np.exp(-1*delta) > zeta:
            accept = True
        else:
            accept = False
    
    return accept

def pressure_delta(density, r_cut):
    sr3 = 1/(r_cut**3)
    pressure= (math.pi)*(8/3)*(sr3**3 - sr3)*(density**2)
    return pressure

def pbc(r):
    for i in range(0, np.size(r,0)):
        for j in range(0, np.size(r, 1)):
            r[i, j] = r[i, j] - int(r[i,j])
    
    return r

def pbc_1(r):
    for i in range(0, np.size(r)):
        r[i] = r[i] - int(r[i])
    return r

def printdata(variables, fff):
    #[m1_r, m2_r, x12_r, x21_r, v_r, n_1, n_2, density_1, density_2, e1_c, e2_c, p1_c, p2_c]
    print('  Averages of quantities in Boxes   ', file = fff)
    print('m1_r:  ', variables[0], '  m2_r:  ', variables[1], file = fff)
    print('x12_r:  ', variables[2], '  x21_r:  ', variables[3], file = fff)
    print('v_r:  ', variables[4], file = fff)
    print('Temperature of the system:   ', temperature, file = fff)
    print('Number of molecules in Boxes:    ', file = fff)
    print('Box 1:  ', variables[5], '  Box 2:  ', variables[6], file = fff)
    print('Density of Boxes:  ', file = fff)
    print('Box 1:  ', variables[7], '  Box 2:  ', variables[8], file = fff)
    print('Energy of Boxes:  ', file = fff)
    print('Box 1:  ', variables[9], '  Box 2:  ', variables[10], file = fff)
    print('Pressure of the Boxes:  ', file = fff)
    print('Box1:  ', variables[11], '  Box 2:  ', variables[12], file = fff)
    return
#@jit(target_backend='cuda')
def atomdisp(n, r, box, r_cut, dr_max, temperature, total, m_acc, m1_ratio, m2_ratio):
    
            # Atom Displacement Moves in Box 1 and Box 2
    for i in range(0, int(n[0])):    # Loop over Atoms in system 1
        partial_old = potential_1(0, n[0], r[:,i], i, box[0], r_cut)
        # Old Atom Potential, Virial
        if partial_old[2] == False :
            #exit('Overlap in Current Configuration in Box 1')
            continue
        ri[:] = random_translate_vector(dr_max/box[0], r[:,i])
        ri[:] = pbc_1(ri)
        
        partial_new = potential_1(0, n[0], ri, i, box[0], r_cut)
            
        if partial_new[2] == False :
            delta = partial_new[0] - partial_old[0]
            delta = delta/temperature
                
            if metropolis(delta) == True:
                move(i, ri, r)
                total[0] = total[0] + partial_new - partial_old
                m_acc = m_acc + 1
    
    m1_ratio = m_acc/n[0]
    m_acc = 0.0
    
    for i in range(int(n[0]), int(n[0]+n[1])):  # Loop over atoms in System 2
        partial_old = potential_1(n[0], n[0]+n[1], r[:,i], i, box[1], r_cut)
        # Old Atom Potential, Virial
        if partial_old[2] == True :
            #exit('Overlap in Current Configuration in Box 2')
            continue
            
        ri[:] = random_translate_vector(dr_max/box[1], r[:,i])
        ri[:] = pbc_1(ri)
            
        partial_new = potential_1(n[0], n[0]+n[1], ri, i, box[1], r_cut)
    
        if partial_new[2] == False :
            delta = partial_new[0] - partial_old[0]
            delta = delta/temperature
            
            if metropolis(delta) == True:
                move(i, ri, r)
                total[1] = total[1] + partial_new - partial_old
                m_acc = m_acc + 1
        
    m2_ratio = m_acc/n[1]
    return total, m1_ratio, m2_ratio
#@jit(target_backend='cuda')
def atomexchange(nswap, x12_try, n, r, box, r_cut, total, x12_acc, x21_try, x21_acc, temperature):
            # Atom Exchange Moves
    for iswap in range(1, nswap):
        ri = np.random.uniform(0,1,3)
        ri = ri -0.5
        zeta = np.random.uniform(0,1)
              
        if zeta > 0.5:               # Try swapping 1 -> 2
            x12_try = x12_try + 1
            if n[0] > 1:
                i = np.random.randint(0, n[0])
                partial_old = potential_1(0, n[0]-1, r[:,i], i, box[0], r_cut)
                if partial_old[2] == True :
                    #exit('Overlap in current confirguration in box 1')
                    continue
                
                partial_new = potential_1(n[0], n[0]+n[1], ri, 0, box[1], r_cut)
                
                if partial_new[2] == False :
                    delta = ( partial_new[0] - partial_old[0]) / temperature 
                    delta = delta - np.log( box[1]**3 / (n[1]+1) ) 
                    delta = delta + np.log( box[0]**3 / (n[0]) )   
                    
                    if metropolis(delta) == True:
                        swap(i, ri, r, n)
                        total[0] = total[0] - partial_old
                        total[1] = total[1] - partial_new
                        x12_acc = x12_acc + 1
            
        else:
            x21_try = x21_try + 1
                    
            if n[1] > 1:
                i = np.random.randint(n[0], n[0]+n[1])
                partial_old = potential_1(n[0], n[0]+n[1], r[:,i], i, box[1], r_cut)
                if partial_old[2] == True:
                    #exit('Overlap in current configuration in Box 2')
                    continue
                    
                partial_new = potential_1(1, n[0]-1, ri, 0, box[0], r_cut)
                if partial_new[2] == False:
                    delta = (partial_new[0] - partial_old[0])/temperature
                    delta = delta - np.log((box[0]**3)/(n[0]+1))
                    delta = delta + np.log((box[1]**3)/n[1])
                        
                    if metropolis(delta) == True:
                        swap(i, ri, r, n)
                        total[1] = total[1] - partial_old
                        total[0] = total[0] + partial_new
                        x21_acc = x21_acc + 1
    x12_ratio = 0
    if x12_try > 0: x12_ratio = x12_acc/x12_try
    x21_ratio = 0
    if x21_try > 0: x21_ratio = x21_acc/x21_try
    return total, x12_ratio, x21_ratio
#@jit(target_backend='cuda')
def vol_rearr(n_vol, dv_max, box, r_cut, n, total, temperature, ):
            # Volume Rearrangement Move
    for v in range(0, n_vol):
        v_ratio = 0
        zeta = random.random()
        dv = dv_max*((2*zeta) - 1)
        vol_old[:] = box[:]**3
        vol_new[:] = vol_old[:] + [-dv , dv]
        box_new[:] = vol_new[:]**(1.0/3.0)
        if np.min(box_new) < 2.0*r_cut:
            sys.exit('Box length too small')
        
        total_new[0] = potential(0, n[0]-1, box_new[0], r_cut)
        total_new[1] = potential(n[0], n[1]+n[0], box_new[1], r_cut)
        
        if total_new[0,2] == False and total_new[1,2] == False :
            delta = np.sum(total_new[:,0]) - np.sum(total[:,0])
            delta = delta/temperature
            delta = delta - (n[0]*np.log(vol_new[0]/vol_old[0]))
            delta = delta - (n[1]*np.log(vol_new[1]/vol_old[1]))
            
            if metropolis(delta) == True:
                total[:] = total_new[:]
                box[:] = box_new[:]
                v_ratio = 1
    return v_ratio, total, box

    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#sourceFile = open('T_0.3.txt', 'w+')
blk_file = open('Blk_avg_T_130.txt', 'w+')
dens_file = open('Density_T_130.txt', 'w+')
ener_file = open('Energy_T_130.txt', 'w+')
press_file = open('Pressure_T_130.txt', 'w+')

print('Density of Boxes Through all steps', file = dens_file)
print('Step','    ','Box 1','    ','Box 2', file = dens_file)
print('Energy of Boxes Through all steps', file = ener_file)
print('Step','    ','Box 1','    ','Box 2', file = ener_file)
print('Pressure of Boxes Through all steps', file = press_file)
print('Step','    ','Box 1','    ','Box 2', file = press_file)

# Main code

N = 500                  # Total Number of particles in 2 boxes

box = np.zeros(2)
Density = np.zeros(2)
total = np.zeros((2,3))
total_new = np.zeros((2,3))
n = np.zeros(2)

nh = int(300)
rho_min = 0.0
rho_max = 0.9
rho_del = (rho_max - rho_min)/nh
eng_min = -3.3
eng_max = 1.2
eng_del = (eng_max - eng_min)/nh

#rho_hist = np.zeros(nh)
#eng_hist = np.zeros(nh)

ri = np.zeros(3)
box_new = np.zeros(2)
vol_new = np.zeros(2)
vol_old = np.zeros(2)

introduction()
print('Gibbs Ensemble MonteCarlo', file = blk_file)

nblock = 200
nstep = 100
nswap = 50
n_vol = 1
temperature = 1.3
#temperature = (kB * temperature)/epsilon
r_cut = 2.5
dr_max = 0.15
dv_max = 10.0

print('Number of Blocks:   ', nblock, file = blk_file)
print('Number of Steps per Block:  ', nstep, file = blk_file)
print('Swap attempts per step:  ', nswap, file = blk_file)
print('Temperature:  ', temperature, file = blk_file)
print('Potential cutoff distance:  ', r_cut, file = blk_file)
print('Maximum Displacement:  ', dr_max, file = blk_file)
print('Maximum Volume change:  ', dv_max, file = blk_file)

# Setting up the 2 boxes
n[0] = 250
n[1] = 250

r = np.zeros((3,int(n[0] + n[1])))

Density[0] = 0.345
Density[1] = 0.345

#Density[0] = Density[0]*(sigma**3)
#Density[1] = Density[1]*(sigma**3)

box[0] = (n[0]/(Density[0]))**(1/3)
box[1] = (n[1]/(Density[1]))**(1/3)
print('Length of Box 1:  ', box[0], file = blk_file)
print('Length of Box 2:  ', box[1], file = blk_file)

r[0, 0:int(n[0])], r[1, 0:int(n[0])], r[2, 0:int(n[0])] = Lattice(n[0], box[0])
r[0, int(n[0]):int(n[0]+n[1])], r[1, int(n[0]):int(n[0]+n[1])], r[2, int(n[0]):int(n[0]+n[1])] = Lattice(n[1], box[1])

#allocate_arrays(box, r_cut)

r[:, 0:int(n[0])] = r[:, 0:int(n[0])]/box[0]
r[:, int(n[0]):int(n[0]+n[1])] = r[:, int(n[0]):int(n[0]+n[1])]/box[1]

r = pbc(r)   # PBC

# Initial Energy and Overlap Check
total[0] = potential(0, n[0], box[0], r_cut)
total[1] = potential(n[0], n[0]+n[1], box[1], r_cut)

#if total[0,2] == True : 
#    exit('Overlap in initial Configuration of Box 1 !')

#if total[1,2] == True :
#    exit('Overlap in Initial Configuration of Box 2 !')

# Initialize arrays for averaging 
m1_ratio = 0.0
m2_ratio = 0.0
x12_ratio = 0.0
x21_ratio = 0.0
v_ratio = 0.0

run_avg, run_err, run_nrm = run_begin()

# Zeros Histograms
#rho_hist[:] = 0.0
#eng_hist[:] = 0.0

print('#----------------------------------------------------------------------------------------------------------------------------------------------------#', file = blk_file)
print('Initial Condition:  ', file = blk_file)
printdata(calc_variables(box, n, m1_ratio, m2_ratio, x12_ratio, x21_ratio, v_ratio, temperature, total, r_cut), blk_file)

for blk in range(1, nblock):
    blk_avg, blk_msd, blk_nrm = blk_begin()
    for stp in range(1, nstep):
        #w = random.random()*3
        #if w < float(1):
        m_acc = 0.0 
        total, m1_ratio, m2_ratio = atomdisp(n, r, box, r_cut, dr_max, temperature, total, m_acc, m1_ratio, m2_ratio)   
            			     
        #elif w> float(1) and w< float(2):
        x12_try = 0
        x12_acc = 0 
        x21_try = 0
        x21_acc = 0	
        total, x12_ratio, x21_ratio = atomexchange(nswap, x12_try, n, r, box, r_cut, total, x12_acc, x21_try, x21_acc, temperature)	

        #else:
        v_ratio, total, box = vol_rearr(n_vol, dv_max, box, r_cut, n, total, temperature)

        
        #print('#----------------------------------------------------------------------------------------------------------------------------------------------------#', file = sourceFile)
        #print('After step :  ', stp,'  in block:  ', blk, file = sourceFile)
        #printdata(calc_variables(box, n, m1_ratio, m2_ratio, x12_ratio, x21_ratio, v_ratio, temperature, total, r_cut), sourceFile)
        
        blk_avg, blk_msd, blk_nrm = blk_add(calc_variables(box, n, m1_ratio, m2_ratio, x12_ratio, x21_ratio, v_ratio, temperature, total, r_cut), blk_avg, blk_msd, blk_nrm)
        #add_hist(n, box, rho_min, rho_del, rho_hist, total, nh, eng_min, eng_del)
        var = calc_variables(box, n, m1_ratio, m2_ratio, x12_ratio, x21_ratio, v_ratio, temperature, total, r_cut)
        print(stp,'    ',var[7],'    ',var[8], file=dens_file)
        print(stp,'    ',var[9],'    ',var[10],file = ener_file )
        print(stp,'    ',var[11],'    ',var[12], file = press_file)
    #print('#----------------------------------------------------------------------------------------------------------------------------------------------------#', file = sourceFile)
    #print(' After Block Move:   ', blk, file = blk_file)
    #printdata(calc_variables(box, n, m1_ratio, m2_ratio, x12_ratio, x21_ratio, v_ratio, temperature, total, r_cut), blk_file)
    #plot3(r)
    
    blk_avg, blk_msd, blk_nrm, run_avg, run_err, run_nrm = blk_end(blk_avg, blk_msd, blk_nrm, run_avg, run_err, run_nrm)
    #if nblock< 1000: print(blk, file = blk_file)
    print('#----------------------------------------------------------------------------------------------------------------------------------------------------#', file = blk_file)
    print(' After Block Move:   ', blk, file = blk_file)
    printdata(blk_avg, blk_file)

run_avg, run_err, run_nrm = run_end(calc_variables(box, n, m1_ratio, m2_ratio, x12_ratio, x21_ratio, v_ratio, temperature, total, r_cut), run_avg, run_err, run_nrm)
print('#----------------------------------------------------------------------------------------------------------------------------------------------------#', file = blk_file)
print('After GEMC runs:   ', file = blk_file)
#printdata(run_avg, blk_file)
printdata(run_avg, blk_file)
print('Length of Box 1:  ', box[0], file = blk_file)
print('Length of Box 2:  ', box[1], file = blk_file)
dens_file.close()
ener_file.close()
press_file.close()
with open('Final_output_T_130.txt', 'w') as f:
    with redirect_stdout(f):
        printdata(run_avg, f)
        print('Coordinates of atoms:  ')
        for i in range(0,int(n[0]+n[1])):
            print(i,':  ',r[:,i])
blk_file.close()       
#write_hist()
#deallocate_arrays()
conclusion()


                  
                
            
# %%

