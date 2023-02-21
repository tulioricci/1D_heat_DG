import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time
import pickle

from discretization import *
from filterfuncs import *

np.set_printoptions(edgeitems=10,linewidth=132,suppress=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Ng = 11
Np = 2

tfinal = 10.0
#integration = "SSPRK3"
integration = "Euler"

CFL = 0.15/(2.0*Np + 3)

#============================

solid_discr = discretization( np.linspace( 0.00, 0.02, 20+1), Np+1)
fluid_discr = discretization( np.linspace( 0.02, 0.04, 20+1), Np+1)

#============================

fluid_kappa = 0.00125
solid_kappa = fluid_kappa*10.0

#============================

x_fluid = fluid_discr[0]
dx = x_fluid[-1,:] - x_fluid[0,:]
dt_fluid = np.min(CFL*dx**2/fluid_kappa)

x_solid = solid_discr[0]
dx = x_solid[-1,:] - x_solid[0,:]
dt_solid = np.min(CFL*dx**2/solid_kappa)

dt = np.minimum(dt_fluid, dt_solid)

print(dt)

#============================

#x = np.array([ x_fluid, x_solid ], dtype=object)
x = [ x_fluid, x_solid ] 

if True:
    u = np.zeros((2), dtype=object)
    u[0] = x[0]*0.0 + 1000.0
    u[1] = x[1]*0.0 + 1000.0
    step = 0
    t = 0.0
else:
    with open('./output/cv-000282000.pickle', 'rb') as handle: 
        step, t, u = pickle.load(handle)
    print('Starting step: ', step)
    print('Starting time: ', t)

#============================

#kappa_interface = 2.0*fluid_kappa*solid_kappa/( fluid_kappa + solid_kappa )
#print(kappa_interface)

#============================

try:
    os.mkdir('./figs')
except:
    print('"figs" already exist')

try:
    os.mkdir('./output')
except:
    print('"output" already exist')    

#============================


import numpy as np

def grad(u, discretization, bc, domain):

    x, Jac, invJac, Dr, Vand, Surf = discretization
    
    Np = x.shape[0]
    Ng = x.shape[1]

    jump_L_u = np.zeros(Ng,)
    jump_R_u = np.zeros(Ng,)
    du    = np.zeros((Np,Ng)) #note that "du" is a full matrix now
    u_0   = np.zeros(Ng,)
    u_N   = np.zeros(Ng,)

    nhat_0 = -1.0 # normal vector on the left will point to the left (-)
    nhat_N = +1.0 # normal vector on the right will point to the right (+)

    u_0[:] = u[ 0,:]  # first point, starting from the second element
    u_N[:] = u[-1,:]  # last point, starting from the first element

    # compute the jump at internal interfaces
    jump_L_u[1:Ng-0] = u_0[1:Ng-0]*nhat_0 + u_N[0:Ng-1]*nhat_N #left face
    jump_R_u[0:Ng-1] = u_0[1:Ng  ]*nhat_0 + u_N[0:Ng-1]*nhat_N #right face

    # apply Dirichlet boundary conditions and evaluate the jump at domain bnds
    if domain == 'fluid':
        u_inlet  = bc[0]
        u_outlet = bc[1]
    if domain == 'solid':
        u_inlet  = u_0[0]
        u_outlet = bc[1]

    jump_L_u[ 0] = u_0[ 0 ]*nhat_0 + u_inlet*nhat_N
    jump_R_u[-1] = u_outlet*nhat_0 + u_N[-1]*nhat_N

    du[ 0,:] = +jump_L_u
    du[-1,:] = +jump_R_u
    
    # end the auxiliary equation
    return invJac*np.matmul(Dr,u) - (Surf)*(du)/2.0
 
#============================

def rhs_viscous(q, discretization, bc, domain):

    x, Jac, invJac, Dr, Vand, Surf = discretization

    Np = x.shape[0]
    Ng = x.shape[1]

    dq    = np.zeros((Np,Ng)) #note that "dq" is a full matrix now

    jump_L_q = np.zeros(Ng,)
    jump_R_q = np.zeros(Ng,)

    q_0   = np.zeros(Ng,)
    q_N   = np.zeros(Ng,)

    nhat_0 = -1.0 # normal vector on the left will point to the left (-)
    nhat_N = +1.0 # normal vector on the right will point to the right (+)

    # start the gradient equation per se
    q_0[:] = q[ 0,:]  # first point, starting from the second element
    q_N[:] = q[-1,:]  # last point, starting from the first element

    # compute the jump at internal interfaces
    jump_L_q[1:Ng-0] = q_0[1:Ng-0]*nhat_0 + q_N[0:Ng-1]*nhat_N #left face
    jump_R_q[0:Ng-1] = q_0[1:Ng  ]*nhat_0 + q_N[0:Ng-1]*nhat_N #right face

    # apply boundary conditions and evaluate the jump at domain bnds
    if domain == 'solid':
        q_inlet  = 0.0
        q_outlet = bc

    if domain == 'fluid':
        q_inlet  = bc
        q_outlet = q_N[-1]

    jump_L_q[ 0] = q_0[ 0 ]*nhat_0 + q_inlet*nhat_N
    jump_R_q[-1] = q_outlet*nhat_0 + q_N[-1]*nhat_N

    dq[ 0,:] = jump_L_q
    dq[-1,:] = jump_R_q

    rhs = invJac*np.matmul(Dr,q) - (Surf)*(dq)/2.0

    return rhs


def source_terms(u, discretization, idx):
    return 1.0*5.68*1e-8*u[idx, idx]**4




timing1 = 0.0
timing2 = 0.0

kk = step

begin = time.time()
while t < tfinal:

    kk += 1

    if kk % 1000 == 0:
        plt.plot(x[0], u[0])
        plt.plot(x[1], u[1])
        plt.ylim(  280, 620)
        plt.xlim(-1.0, 1.0)
        plt.savefig(f'./figs/T-{kk:09d}.png')
        plt.close()

        filename = 'cv-' + str('%09d' % kk)
        with open('./output/' + filename + '.pickle', 'wb') as handle:
            pickle.dump([kk, t, u], handle, protocol=pickle.HIGHEST_PROTOCOL)

    if (integration == "Euler"):

        fluid_t = u[0]
        solid_t = u[1]

        aux1 = time.time()

        fluid_bc = np.array([ solid_t[-1,-1], 1000.0 ])
        solid_bc = np.array([    0.0, fluid_t[ 0, 0] ])
        fluid_grad = grad(fluid_t, fluid_discr, fluid_bc, 'fluid')
        solid_grad = grad(solid_t, solid_discr, solid_bc, 'solid')

        aux2 = time.time()
        timing1 = timing1 + aux2 - aux1

        radiation = source_terms(solid_t, solid_discr, -1)
        fluid_bc = solid_kappa*solid_grad[-1,-1] + radiation
        solid_bc = fluid_kappa*fluid_grad[ 0, 0] - radiation
        fluid_rhs = rhs_viscous(fluid_kappa*fluid_grad, fluid_discr, fluid_bc, 'fluid')
        solid_rhs = rhs_viscous(solid_kappa*solid_grad, solid_discr, solid_bc, 'solid')

        aux3 = time.time()
        timing2 = timing2 + aux3 - aux2

        rhs = np.zeros((2),dtype=object)
        rhs[0] = fluid_rhs
        rhs[1] = solid_rhs
        u  = u  + dt*rhs;

    if (kk % 100 == 0):
        print(kk, t/tfinal, end='\r')
  
    t = t + dt

end = time.time()

print(' ')
print('niter =', kk)
print('Grad time = ', timing1)
print('RHS time = ', timing2)
print('total time = ', end - begin)

plt.plot(x[0],u[0])
plt.plot(x[1],u[1])
plt.xlim(0.0,0.04)

print()
print('fluid:')
print(fluid_t[ 0, 0], fluid_t[-1,-1])
print(fluid_kappa, fluid_grad[ 0, 0])
print('solid:')
print(solid_t[ 0, 0], solid_t[-1,-1])
print(solid_kappa, solid_grad[-1,-1])
print('interface:')
radiation = source_terms(solid_t, solid_discr, -1)
fluid = fluid_kappa*fluid_grad[ 0, 0]
solid = solid_kappa*solid_grad[-1,-1] + radiation
print('radiation =', radiation)
print(solid, fluid)
plt.savefig('T.png',dpi=300)

plt.show()


  
