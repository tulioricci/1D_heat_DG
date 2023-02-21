
import basis
import numpy as np
#import scipy.special

def discretization(xRef, N):

    phi = basis.LegendrePolynomials(N)
    LGL = basis.GaussLobatto(N,phi)
    Vand, invVand, Dr = basis.VandermondeMatrices(N,phi,LGL)

    x, xF = basis.mapping(xRef,LGL)
    dx =  xF[-1,:] - xF[0,:]
    xm = (xF[-1,:] + xF[0,:])*0.5
    nhat = basis.normal_vectors(xF)
    Jac, invJac = basis.Jacobian(x, Dr)

#    lift = basis.Lift1D(N,Vnd)
#    Fscale = 1.0/np.vstack((Jac[0,:],Jac[-1,:]))

    Surf = Vand@Vand.T@invJac

    return x, Jac, invJac, Dr, Vand, Surf
