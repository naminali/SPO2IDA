#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros


# In[2]:


#Functions


# V = Based Shear,R = roof displacement, M=mass, Phi = first mode shape ordinate 
# the result of this function is Forse(F) and Displasment(Delta) 
def MDOF2SDOF(Vb,R,M, Phi):
    EffectiveM = sum(a * b for a, b in zip(M, Phi))
    Sig_MPhi2 = sum(a * b * b for a, b in zip(M, Phi))
    Gama = EffectiveM/Sig_MPhi2
    F = Vb/Gama
    Delta = R/Gama
    return F, Delta

#find intersection
def find_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("Lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


# In[3]:


#Equivalent SDOF 
#Manipolated Strong diagram



Vb1 = [0, 0.34, 0.34, 0.15, 0]  # Array of Base Shear coefficients - main
Del1 = [0, 0.03, 0.07, 0.08, 0.21]  # Array of Roof displacements in meters - main

Vb2 = [0, 0.08, 0.08, 0]  # Array of Base Shear coefficients - w/Mechanism
Del2 = [0, 0.11, 0.15, 0.22]  # Array of Roof displacements in meters - w/Mechanism

# Plotting the main
plt.plot(Del1, Vb1, marker='o', label='Main(strong)')

# Plotting the w/Mechanism
plt.plot(Del2, Vb2, marker='o', label='w/Mechanism')

# Adding labels and title
plt.xlabel('Roof Displacement (m)')
plt.ylabel('Base Shear Coefficient')
plt.title('Infilled frame (strong) 6 Story structure')

# Adding a legend
plt.legend()

# Displaying the plot
plt.show()


# In[4]:


#SDOF of strong

Phi = [0, 0.09, 0.2, 0.33, 0.53,0.77,1]  # first mode shape ordinate

# M is Mass and I assume 800 ton for each storey
M = [800000, 800000, 800000, 800000, 800000, 800000]
#main
Vb = [0, 0.34, 0.34, 0.15, 0]  # Array of Base Shear coefficients - main
R = [0, 0.03, 0.07, 0.08, 0.21]  # Array of Roof displacements in meters - main

F = zeros(len(Vb))#Forse - main
Delta = zeros(len(R))# Displacement - main

for i in range(len(Vb)):
    F[i] = MDOF2SDOF(Vb[i], R[i], M, Phi)[0]
    Delta[i] = MDOF2SDOF(Vb[i], R[i], M, Phi)[1]

    
#w/Mechanism
Vb2 = [0, 0.08, 0.08, 0]  # Array of Base Shear coefficients - w/Mechanism
R2 = [0, 0.11, 0.15, 0.22]  # Array of Roof displacements in meters - w/Mechanism

F2 = zeros(len(Vb2))#Forse - main
Delta2 = zeros(len(R2))# Displacement - main 
for i in range(len(Vb2)):
    F2[i] = MDOF2SDOF(Vb2[i], R2[i], M, Phi)[0]
    Delta2[i] = MDOF2SDOF(Vb2[i], R2[i], M, Phi)[1]

    
#Plot SDOF of strong infilled structure

# Plotting the main
plt.plot(Delta, F, marker='o', label='Main(strong)')

# Plotting the w/Mechanism
plt.plot(Delta2, F2, marker='o', label='w/Mechanism')



# Adding labels and title
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.title('SDOF (strong) 6 Story structure')

# Adding a legend
plt.legend()

# Displaying the plot
plt.show()


# In[5]:


#mu-R strong


# Find Intersection
line1 = ((Delta[2], F[2]), (Delta[3], F[3]))
line2 = ((Delta2[1], F2[1]), (Delta2[2], F2[2]))
intersection = find_intersection(line1, line2)

#Combination of two graphs

Delta3 = zeros(6)
Delta3[0:3] =Delta[0:3]
Delta3[3] = intersection[0]
Delta3[4:6] = Delta2[2:4]

F3 = zeros(6)
F3[0:3] =F[0:3]
F3[3] = intersection[1]
F3[4:6] = F2[2:4]

#Yilding F and Delta
Fy = F3[1]
Deltay = Delta3[1]

#R and Mu
r = F3/Fy
Mu = Delta3/Deltay

# Plot the back bone diagram
plt.plot(Mu, r, marker='o', label='Strong Backbone')


# Adding labels and title
plt.xlabel('Mu')
plt.ylabel('R')
plt.title('Backbone of strong infilled 6 Story structure')

# Adding a legend
plt.legend()

# Displaying the plot
plt.show()


# In[6]:


#IDA for strong

T = 0.41
mu = Mu[2:]
R = r[2:]

# Define the coefficients {16%, 50%, 84%}
# Hardening branch
a_alpha_1 = np.array([[0.146, 0.8628, 1.024],
                     [0.5926, 0.9235, 0.6034],
                     [0.07312, 0.9195, 0.2466],
                     [0.2965, 0.9632, 0.06141],
                     [0.02688, 0.4745, 0.2511],
                     [1.063, 0.0654, 0.0001],
                     [0.3127, 0.04461, 0.07086]])

b_alpha_1 = np.array([[0.5335, 0.7624, 0.9018],
                     [0.4161, 0.5041, 0.1928],
                     [0.4495, 0.1785, 0.4758],
                     [0.2215, 1.022, 0.6903],
                     [0.3699, 0.3253, 0.3254],
                     [1.003, 0.4064, 0.939],
                     [0.1462, 0.4479, 0.3948]])

c_alpha_1 = np.array([[0.03444, 0.1643, 0.6555],
                     [0.3194, 0.1701, 0.1072],
                     [0.01667, 0.1147, 0.1232],
                     [0.1087, 0.1694, 0.05664],
                     [0.0158, 0.09403, 0.07067],
                     [0.646, 0.02054, 0.00132],
                     [0.07181, 0.01584, 0.02287]])

a_beta_1 = np.array([[0.2008, -0.1334, 0.7182],
                    [0.179, 0.3312, 0.132],
                    [0.1425, 0.7985, 0.1233],
                    [0.1533, 0.0001, 0.09805],
                    [3.623E+12, 0.1543, 0.1429],
                    [0.09451, 0.9252, 0.6547],
                    [0.1964, 0.2809, 0.0001]])

b_beta_1 = np.array([[1.093, 0.7771, 0.04151],
                    [0.7169, 0.7647, 0.6058],
                    [0.4876, 0.04284, 0.4904],
                    [0.5709, 0.5721, 0.5448],
                    [97.61, 0.4788, 0.3652],
                    [0.4424, 0.8165, 0.8431],
                    [0.3345, 0.3003, 0.7115]])

c_beta_1 = np.array([[0.5405, 0.04907, 0.09018],
                    [0.08836, 0.000986, 0.04845],
                    [0.04956, 0.09365, 0.04392],
                    [0.07256, 0.0001, 0.01778],
                    [17.94, 0.105, 0.09815],
                    [0.06262, 0.51, 0.7126],
                    [0.09522, 0.1216, 0.0001803]])

# Softening branch
a_alpha_2 = np.array([0.03945, 0.01833, 0.009508])
b_alpha_2 = np.array([-0.03069, -0.01481, -0.007821])
a_beta_2 = np.array([1.049, 0.8237, 0.4175])
b_beta_2 = np.array([0.2494, 0.04082, 0.03164])
a_gamma_2 = np.array([-0.7326, -0.7208, -0.0375])
b_gamma_2 = np.array([1.116, 1.279, 1.079])

# Residual plateau branch
a_alpha_3 = np.array([-5.075, -2.099, -0.382])
b_alpha_3 = np.array([7.112, 3.182, 0.6334])
c_alpha_3 = np.array([-1.572, -0.6989, -0.051])
d_alpha_3 = np.array([0.1049, 0.0481, 0.002])

a_beta_3 = np.array([16.16, 8.417, -0.027])
b_beta_3 = np.array([-26.5, -14.51, -1.80])
c_beta_3 = np.array([10.92, 6.75, 2.036])
d_beta_3 = np.array([1.055, 0.9061, 1.067])

# Strength degradation branch
a_alpha_4 = np.array([-1.564, -0.5954, -0.06693])
b_alpha_4 = np.array([2.193, 0.817, 0.1418])
c_alpha_4 = np.array([-0.352, -0.09191, 0.0124])
d_alpha_4 = np.array([0.0149, 0.001819, -0.002012])
a_beta_4 = np.array([1.756, 0.7315, -0.408])
b_beta_4 = np.array([-8.719, -3.703, -1.333])
c_beta_4 = np.array([8.285, 4.391, 2.521])
d_beta_4 = np.array([1.198, 1.116, 1.058])

# Compute the parameters
# For each fractile i in 16%, 50%, and 84%
alpha_1 = np.zeros(3)
beta_1 = np.zeros(3)
alpha_2 = np.zeros(3)
beta_2 = np.zeros(3)
gamma_2 = np.zeros(3)
alpha_3 = np.zeros(3)
beta_3 = np.zeros(3)
alpha_4 = np.zeros(3)
beta_4 = np.zeros(3)

for i in range(3):
    # Hardening branch
    alpha_1[i] = np.sum(a_alpha_1[:, i] * np.exp(-((T - b_alpha_1[:, i]) / c_alpha_1[:, i]) ** 2))
    beta_1[i] = np.sum(a_beta_1[:, i] * np.exp(-((T - b_beta_1[:, i]) / c_beta_1[:, i]) ** 2))

    # Softening branch
    alpha_2[i] = a_alpha_2[i] * T + b_alpha_2[i]
    beta_2[i] = a_beta_2[i] * T + b_beta_2[i]
    gamma_2[i] = a_gamma_2[i] * T + b_gamma_2[i]

    # Residual branch
    alpha_3[i] = a_alpha_3[i] * T ** 3 + b_alpha_3[i] * T ** 2 + c_alpha_3[i] * T + d_alpha_3[i]
    beta_3[i] = a_beta_3[i] * T ** 3 + b_beta_3[i] * T ** 2 + c_beta_3[i] * T + d_beta_3[i]

    # Strength degradation branch
    alpha_4[i] = a_alpha_4[i] * T ** 3 + b_alpha_4[i] * T ** 2 + c_alpha_4[i] * T + d_alpha_4[i]
    beta_4[i] = a_beta_4[i] * T ** 3 + b_beta_4[i] * T ** 2 + c_beta_4[i] * T + d_beta_4[i]

# Fit the branches
# Initialize some arrays
mu_1 = np.linspace(1, mu[0], 10)  # ductility in hardening branch
mu_2 = np.linspace(mu[0], mu[1], 10)  # ductility in softening branch
mu_3 = np.linspace(mu[1], mu[2], 10)  # ductility in residual plateau branch
mu_4 = np.linspace(mu[2], mu[3], 10)  # ductility in degradation branch

# Fit & adjust the discontinuities between the branches
Rdyn_1 = np.zeros((3, len(mu_1)))
Rdyn_2 = np.zeros((3, len(mu_2)))
Rdyn_3 = np.zeros((3, len(mu_3)))
Rdyn_4 = np.zeros((3, len(mu_4)))

for j in range(3):
    for i in range(len(mu_1)):
        Rdyn_1[j, i] = alpha_1[j] * mu_1[i] ** beta_1[j]  # fit the hardening branch

for j in range(3):
    for i in range(len(mu_2)):
        Rdyn_2[j, i] = alpha_2[j] * mu_2[i] ** 2 + beta_2[j] * mu_2[i] + gamma_2[j]  # fit the softening branch

for j in range(3):
    for i in range(len(mu_3)):
        Rdyn_3[j, i] = alpha_3[j] * mu_3[i] + beta_3[j]  # fit the residual strength branch

for j in range(3):
    for i in range(len(mu_4)):
        Rdyn_4[j, i] = alpha_4[j] * mu_4[i] + beta_4[j]  # fit the degradation branch

Rdyn = np.concatenate((Rdyn_1, Rdyn_2, Rdyn_3, Rdyn_4), axis=1).T
mudyn = np.concatenate((mu_1, mu_2, mu_3, mu_4))

# Hardening Initiation
Rdiff0 = 1 - Rdyn[0, :3]
Rdyn[:10, :3] += Rdiff0

# Connection Hardening-Softening
Rdiff1 = Rdyn[9, :3] - Rdyn[10, :3]
Rdyn[10:20, :3] += Rdiff1

# Connection Softening-Plateau
Rdiff2 = Rdyn[19, :3] - Rdyn[20, :3]
Rdyn[20:30, :3] += Rdiff2

# Connection Plateau-Degradation
Rdiff3 = Rdyn[29, :3] - Rdyn[30, :3]
Rdyn[30:40, :3] += Rdiff3

# Add in a flatline point
Rdyn = np.vstack((Rdyn, Rdyn[-1, :]))
mudyn = np.append(mudyn, mudyn[-1] + 5)

# Plot the diagram
plt.figure()
plt.plot(mudyn, Rdyn[:, 0], '-.g')
plt.plot(mudyn, Rdyn[:, 1], '-r')
plt.plot(mudyn, Rdyn[:, 2], '-.m')
plt.plot([0, 1, *mu], [0, 1, *R], '-k')
plt.legend(['84%', '50%', '16%', 'SPO'], loc='southeast')
plt.xlabel('Ductility mu')
plt.ylabel('Strength Ratio R')
plt.grid(True)
plt.show()


# In[7]:


# Plot the diagram that Dr. Zaker asked 
plt.figure()
plt.plot(mudyn, Rdyn[:, 1], '-r')
plt.plot([0, 1, *mu], [0, 1, *R], '-k')
plt.legend(['50%', 'Backbone curve'], loc='southeast')
plt.xlabel('Ductility mu')
plt.ylabel('Strength Ratio R')
plt.grid(True)
plt.show()


# # From here on, I didn't include it in the review, I was just curious to see how it would be for the weak Infill structure.
# 

# In[8]:


#Manipolated weak diagram



import matplotlib.pyplot as plt


Vb1 = [0, 0.17, 0.17, 0.1, 0]  # Array of Base Shear coefficients - main
Del1 = [0, 0.03, 0.09, 0.1, 0.21]  # Array of Roof displacements in meters - main

Vb2 = [0, 0.08, 0.08, 0]  # Array of Base Shear coefficients - w/Mechanism
Del2 = [0, 0.11, 0.15, 0.22]  # Array of Roof displacements in meters - w/Mechanism

# Plotting the main
plt.plot(Del1, Vb1, marker='o', label='Main(weak)')

# Plotting the w/Mechanism
plt.plot(Del2, Vb2, marker='o', label='w/Mechanism')

# Adding labels and title
plt.xlabel('Roof Displacement (m)')
plt.ylabel('Base Shear Coefficient')
plt.title('Infilled frame (weak) 6 Story structure')

# Adding a legend
plt.legend()

# Displaying the plot
plt.show()


# In[9]:


#SDOF of weak

Phi = [0, 0.09, 0.2, 0.33, 0.53,0.77,1]  # first mode shape ordinate

# M is Mass and I assume 800 ton for each storey
M = [800000, 800000, 800000, 800000, 800000, 800000]
#main
Vb = [0, 0.17, 0.17, 0.1, 0]  # Array of Base Shear coefficients - main
R = [0, 0.03, 0.09, 0.1, 0.21]  # Array of Roof displacements in meters - main

F = zeros(len(Vb))#Forse - main
Delta = zeros(len(R))# Displacement - main

for i in range(len(Vb)):
    F[i] = MDOF2SDOF(Vb[i], R[i], M, Phi)[0]
    Delta[i] = MDOF2SDOF(Vb[i], R[i], M, Phi)[1]

    
#w/Mechanism
Vb2 = [0, 0.08, 0.08, 0]  # Array of Base Shear coefficients - w/Mechanism
R2 = [0, 0.11, 0.15, 0.22]  # Array of Roof displacements in meters - w/Mechanism

F2 = zeros(len(Vb2))#Forse - main
Delta2 = zeros(len(R2))# Displacement - main 
for i in range(len(Vb2)):
    F2[i] = MDOF2SDOF(Vb2[i], R2[i], M, Phi)[0]
    Delta2[i] = MDOF2SDOF(Vb2[i], R2[i], M, Phi)[1]

    
#Plot SDOF of strong infilled structure

# Plotting the main
plt.plot(Delta, F, marker='o', label='Main(weak)')

# Plotting the w/Mechanism
plt.plot(Delta2, F2, marker='o', label='w/Mechanism')

# Adding labels and title
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.title('SDOF (weak) 6 Story structure')

# Adding a legend
plt.legend()

# Displaying the plot
plt.show()


# In[10]:


#mu-R weak


# Find Intersection
line1 = ((Delta[2], F[2]), (Delta[3], F[3]))
line2 = ((Delta2[1], F2[1]), (Delta2[2], F2[2]))
intersection = find_intersection(line1, line2)

#Combination of two graphs

Delta3 = zeros(6)
Delta3[0:3] =Delta[0:3]
Delta3[3] = intersection[0]
Delta3[4:6] = Delta2[2:4]

F3 = zeros(6)
F3[0:3] =F[0:3]
F3[3] = intersection[1]
F3[4:6] = F2[2:4]

#Yilding F and Delta
Fy = F3[1]
Deltay = Delta3[1]

#R and Mu
r = F3/Fy
Mu = Delta3/Deltay

# Plot the back bone diagram
plt.plot(Mu, r, marker='o', label='Strong Backbone')


# Adding labels and title
plt.xlabel('Mu')
plt.ylabel('R')
plt.title('Backbone of strong infilled 6 weak structure')

# Adding a legend
plt.legend()

# Displaying the plot
plt.show()


# In[12]:


#IDA for weak

T = 0.41
mu = Mu[2:]
R = r[2:]

# Define the coefficients {16%, 50%, 84%}
# Hardening branch
a_alpha_1 = np.array([[0.146, 0.8628, 1.024],
                     [0.5926, 0.9235, 0.6034],
                     [0.07312, 0.9195, 0.2466],
                     [0.2965, 0.9632, 0.06141],
                     [0.02688, 0.4745, 0.2511],
                     [1.063, 0.0654, 0.0001],
                     [0.3127, 0.04461, 0.07086]])

b_alpha_1 = np.array([[0.5335, 0.7624, 0.9018],
                     [0.4161, 0.5041, 0.1928],
                     [0.4495, 0.1785, 0.4758],
                     [0.2215, 1.022, 0.6903],
                     [0.3699, 0.3253, 0.3254],
                     [1.003, 0.4064, 0.939],
                     [0.1462, 0.4479, 0.3948]])

c_alpha_1 = np.array([[0.03444, 0.1643, 0.6555],
                     [0.3194, 0.1701, 0.1072],
                     [0.01667, 0.1147, 0.1232],
                     [0.1087, 0.1694, 0.05664],
                     [0.0158, 0.09403, 0.07067],
                     [0.646, 0.02054, 0.00132],
                     [0.07181, 0.01584, 0.02287]])

a_beta_1 = np.array([[0.2008, -0.1334, 0.7182],
                    [0.179, 0.3312, 0.132],
                    [0.1425, 0.7985, 0.1233],
                    [0.1533, 0.0001, 0.09805],
                    [3.623E+12, 0.1543, 0.1429],
                    [0.09451, 0.9252, 0.6547],
                    [0.1964, 0.2809, 0.0001]])

b_beta_1 = np.array([[1.093, 0.7771, 0.04151],
                    [0.7169, 0.7647, 0.6058],
                    [0.4876, 0.04284, 0.4904],
                    [0.5709, 0.5721, 0.5448],
                    [97.61, 0.4788, 0.3652],
                    [0.4424, 0.8165, 0.8431],
                    [0.3345, 0.3003, 0.7115]])

c_beta_1 = np.array([[0.5405, 0.04907, 0.09018],
                    [0.08836, 0.000986, 0.04845],
                    [0.04956, 0.09365, 0.04392],
                    [0.07256, 0.0001, 0.01778],
                    [17.94, 0.105, 0.09815],
                    [0.06262, 0.51, 0.7126],
                    [0.09522, 0.1216, 0.0001803]])

# Softening branch
a_alpha_2 = np.array([0.03945, 0.01833, 0.009508])
b_alpha_2 = np.array([-0.03069, -0.01481, -0.007821])
a_beta_2 = np.array([1.049, 0.8237, 0.4175])
b_beta_2 = np.array([0.2494, 0.04082, 0.03164])
a_gamma_2 = np.array([-0.7326, -0.7208, -0.0375])
b_gamma_2 = np.array([1.116, 1.279, 1.079])

# Residual plateau branch
a_alpha_3 = np.array([-5.075, -2.099, -0.382])
b_alpha_3 = np.array([7.112, 3.182, 0.6334])
c_alpha_3 = np.array([-1.572, -0.6989, -0.051])
d_alpha_3 = np.array([0.1049, 0.0481, 0.002])

a_beta_3 = np.array([16.16, 8.417, -0.027])
b_beta_3 = np.array([-26.5, -14.51, -1.80])
c_beta_3 = np.array([10.92, 6.75, 2.036])
d_beta_3 = np.array([1.055, 0.9061, 1.067])

# Strength degradation branch
a_alpha_4 = np.array([-1.564, -0.5954, -0.06693])
b_alpha_4 = np.array([2.193, 0.817, 0.1418])
c_alpha_4 = np.array([-0.352, -0.09191, 0.0124])
d_alpha_4 = np.array([0.0149, 0.001819, -0.002012])
a_beta_4 = np.array([1.756, 0.7315, -0.408])
b_beta_4 = np.array([-8.719, -3.703, -1.333])
c_beta_4 = np.array([8.285, 4.391, 2.521])
d_beta_4 = np.array([1.198, 1.116, 1.058])

# Compute the parameters
# For each fractile i in 16%, 50%, and 84%
alpha_1 = np.zeros(3)
beta_1 = np.zeros(3)
alpha_2 = np.zeros(3)
beta_2 = np.zeros(3)
gamma_2 = np.zeros(3)
alpha_3 = np.zeros(3)
beta_3 = np.zeros(3)
alpha_4 = np.zeros(3)
beta_4 = np.zeros(3)

for i in range(3):
    # Hardening branch
    alpha_1[i] = np.sum(a_alpha_1[:, i] * np.exp(-((T - b_alpha_1[:, i]) / c_alpha_1[:, i]) ** 2))
    beta_1[i] = np.sum(a_beta_1[:, i] * np.exp(-((T - b_beta_1[:, i]) / c_beta_1[:, i]) ** 2))

    # Softening branch
    alpha_2[i] = a_alpha_2[i] * T + b_alpha_2[i]
    beta_2[i] = a_beta_2[i] * T + b_beta_2[i]
    gamma_2[i] = a_gamma_2[i] * T + b_gamma_2[i]

    # Residual branch
    alpha_3[i] = a_alpha_3[i] * T ** 3 + b_alpha_3[i] * T ** 2 + c_alpha_3[i] * T + d_alpha_3[i]
    beta_3[i] = a_beta_3[i] * T ** 3 + b_beta_3[i] * T ** 2 + c_beta_3[i] * T + d_beta_3[i]

    # Strength degradation branch
    alpha_4[i] = a_alpha_4[i] * T ** 3 + b_alpha_4[i] * T ** 2 + c_alpha_4[i] * T + d_alpha_4[i]
    beta_4[i] = a_beta_4[i] * T ** 3 + b_beta_4[i] * T ** 2 + c_beta_4[i] * T + d_beta_4[i]

# Fit the branches
# Initialize some arrays
mu_1 = np.linspace(1, mu[0], 10)  # ductility in hardening branch
mu_2 = np.linspace(mu[0], mu[1], 10)  # ductility in softening branch
mu_3 = np.linspace(mu[1], mu[2], 10)  # ductility in residual plateau branch
mu_4 = np.linspace(mu[2], mu[3], 10)  # ductility in degradation branch

# Fit & adjust the discontinuities between the branches
Rdyn_1 = np.zeros((3, len(mu_1)))
Rdyn_2 = np.zeros((3, len(mu_2)))
Rdyn_3 = np.zeros((3, len(mu_3)))
Rdyn_4 = np.zeros((3, len(mu_4)))

for j in range(3):
    for i in range(len(mu_1)):
        Rdyn_1[j, i] = alpha_1[j] * mu_1[i] ** beta_1[j]  # fit the hardening branch

for j in range(3):
    for i in range(len(mu_2)):
        Rdyn_2[j, i] = alpha_2[j] * mu_2[i] ** 2 + beta_2[j] * mu_2[i] + gamma_2[j]  # fit the softening branch

for j in range(3):
    for i in range(len(mu_3)):
        Rdyn_3[j, i] = alpha_3[j] * mu_3[i] + beta_3[j]  # fit the residual strength branch

for j in range(3):
    for i in range(len(mu_4)):
        Rdyn_4[j, i] = alpha_4[j] * mu_4[i] + beta_4[j]  # fit the degradation branch

Rdyn = np.concatenate((Rdyn_1, Rdyn_2, Rdyn_3, Rdyn_4), axis=1).T
mudyn = np.concatenate((mu_1, mu_2, mu_3, mu_4))

# Hardening Initiation
Rdiff0 = 1 - Rdyn[0, :3]
Rdyn[:10, :3] += Rdiff0

# Connection Hardening-Softening
Rdiff1 = Rdyn[9, :3] - Rdyn[10, :3]
Rdyn[10:20, :3] += Rdiff1

# Connection Softening-Plateau
Rdiff2 = Rdyn[19, :3] - Rdyn[20, :3]
Rdyn[20:30, :3] += Rdiff2

# Connection Plateau-Degradation
Rdiff3 = Rdyn[29, :3] - Rdyn[30, :3]
Rdyn[30:40, :3] += Rdiff3

# Add in a flatline point
Rdyn = np.vstack((Rdyn, Rdyn[-1, :]))
mudyn = np.append(mudyn, mudyn[-1] + 5)

# Plot the diagram
plt.figure()
plt.plot(mudyn, Rdyn[:, 0], '-.g')
plt.plot(mudyn, Rdyn[:, 1], '-r')
plt.plot(mudyn, Rdyn[:, 2], '-.m')
plt.plot([0, 1, *mu], [0, 1, *R], '-k')
plt.legend(['84%', '50%', '16%', 'SPO'], loc='southeast')
plt.xlabel('Ductility mu')
plt.ylabel('Strength Ratio R')
plt.grid(True)
plt.show()

