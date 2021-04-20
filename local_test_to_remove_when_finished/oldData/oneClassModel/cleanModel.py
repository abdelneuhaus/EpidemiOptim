import numpy as np
from odeintw import odeintw
import matplotlib.pyplot as plt
from parameters import *

dSdt = [0,0,0,0]
dE2dt = [0,0,0,0]
dE3dt = [0,0,0,0]
dE4dt = [0,0,0,0]
dIdt = [0,0,0,0]
dV1dt = [0,0,0,0]
dV2dt = [0,0,0,0]


def model(y, t, p1, p2, p3, alpha, kappa, Gamma, rho, omega, delta, A, infect, sigma):
    S, E2, E3, E4, I, V1, V2 = y

    # Susceptible compartments
    dSdt[0] = - sum(p1)*alpha[0]*A[0]*S[0]*infect[0] + omega[1]*S[1] - sigma*rho[0]*S[0] + omega[1]*V1[0]
    dSdt[1] = - sum(p2)*alpha[1]*A[1]*S[1]*infect[0] + omega[2]*S[2] - omega[1]*S[1] - sigma*rho[0]*S[1] + Gamma[1]*I[1] + omega[2]*V1[1]
    dSdt[2] = - (p3[1]+p3[2])*alpha[2]*A[2]*S[2]*infect[0] + omega[3]*S[3] - omega[2]*S[2] - sigma*rho[0]*S[2] + Gamma[2]*I[2] + omega[3]*(V1[2]+V1[3]+sum(V2))
    dSdt[3] = - omega[3]*S[3] - sigma*rho[0]*S[3] + Gamma[3]*I[3]
    
    # Vaccinated compartments
    dV1dt[0] = sigma*rho[0]*S[0] - sigma*rho[0]*V1[0] - sum(p2)*alpha[1]*A[1]*V1[0]*infect[0] - omega[1]*V1[0]
    dV1dt[1] = sigma*rho[0]*S[1] - sigma*rho[0]*V1[1] - (p3[1]+p3[2])*alpha[2]*A[2]*V1[1]*infect[0] - omega[2]*V1[1]
    dV1dt[2] = sigma*rho[0]*S[2] - sigma*rho[0]*V1[2] - omega[3]*V1[2]
    dV1dt[3] = sigma*rho[0]*S[3] - sigma*rho[0]*V1[3] - omega[3]*V1[3]

    dV2dt[0] = sigma*rho[0]*V1[0] - omega[3]*V2[0]
    dV2dt[1] = sigma*rho[0]*V1[1] - omega[3]*V2[1]
    dV2dt[2] = sigma*rho[0]*V1[2] - omega[3]*V2[2]
    dV2dt[3] = sigma*rho[0]*V1[3] - omega[3]*V2[3]

    # Exposed compartments
    dE2dt[1] = p1[0]*alpha[0]*A[0]*S[0]*infect[0] + p2[0]*alpha[1]*A[1]*S[1]*infect[0] + p2[0]*alpha[1]*A[1]*V1[0]*infect[0] - kappa[1]*E2[1]
    dE2dt[2] = p1[1]*alpha[0]*A[0]*S[0]*infect[0] + p2[1]*alpha[1]*A[1]*S[1]*infect[0] + p3[1]*alpha[2]*A[2]*S[2]*infect[0] + p2[1]*alpha[1]*A[1]*V1[0]*infect[0] + p3[1]*alpha[2]*A[2]*V1[1]*infect[0] - kappa[2]*E2[2]
    dE2dt[3] = p1[2]*alpha[0]*A[0]*S[0]*infect[0] + p2[2]*alpha[1]*A[1]*S[1]*infect[0] + p3[2]*alpha[2]*A[2]*S[2]*infect[0] + p2[2]*alpha[1]*A[1]*V1[0]*infect[0] + p3[2]*alpha[2]*A[2]*V1[1]*infect[0] - kappa[3]*E2[3]

    dE3dt[1] = kappa[1]*E2[1] - kappa[1]*E3[1]
    dE3dt[2] = kappa[2]*E2[2] - kappa[2]*E3[2]
    dE3dt[3] = kappa[3]*E2[3] - kappa[3]*E3[3]

    dE4dt[1] = kappa[1]*E3[1] - kappa[1]*E4[1]
    dE4dt[2] = kappa[2]*E3[2] - kappa[2]*E4[2]
    dE4dt[3] = kappa[3]*E3[3] - kappa[3]*E4[3]            

    # Infected compartments
    dIdt[1] = kappa[1]*E4[1] - delta[1]*I[1] - Gamma[1]*I[1]
    dIdt[2] = kappa[2]*E4[2] - delta[2]*I[2] - Gamma[2]*I[2]
    dIdt[3] = kappa[3]*E4[3] - delta[3]*I[3] - Gamma[3]*I[3] 

    return dSdt, dE2dt, dE3dt, dE4dt, dV1dt, dV2dt, dIdt


# Initial conditions
S0 = [population[0]- 10/6 - 1/6,0,0,0]
E20 = [0,0,0,0]
E30 = [0,0,0,0]
E40 = [0,0,0,0]
V10 = [0,0,0,0]
V20 = [0,0,0,0]
I0 = [0, 10/6, 1/6, 0]

y0 = np.array(S0), np.array(E20), np.array(E30), np.array(E40), np.array(V10), np.array(V20), np.array(I0)
Nt0 = np.array(S0) + np.array(E20) + np.array(E30) + np.array(E40) + np.array(I0) + np.array(V10) + np.array(V20)
Nt0 = sum(Nt0).tolist()
k = 0.8
con, A, c = contactModifiersComputation(0, 0, k)
_Xm = np.multiply(np.multiply(np.matrix(beta)+np.matrix(beta), 1), np.matrix(y0[6])).tolist()
Xm = [sum(x) for x in _Xm]
Ym = np.divide(Xm, Nt0)
#sigma = sigma_calculation(step)
sigma = 1e-20
infect = np.multiply(c,Ym).tolist()
t = np.linspace(0,1,2)
ret = odeintw(model, y0, t, args=(p1, p2, p3, alpha, kappa, Gamma, rho, omega, delta, A, infect, sigma))


i4=generate_age_output(ret, 6, 3)
print(i4)
fig, (ax1) = plt.subplots(1)
ax1.plot(i4, '-m', label='I3 - 0:9 ans')
plt.show()