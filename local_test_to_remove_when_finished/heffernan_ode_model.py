import numpy as np
from odeintw import odeintw # odeint doesn't handle matrices
import matplotlib.pyplot as plt
from parameters import *


def model(y, t, p1, p2, p3, alpha, kappa, Gamma, rho, omega, delta, A, infect, sigma):
    S, E2, E3, E4, I, V1, V2 = y

    # For each age groups :
    for n in range(0, 16):

        # Susceptible compartments
        dSdt[n] = - sum(p1[n])*alpha[n][0]*A[n][0]*S[n][0]*infect[n] + omega[n][1]*S[n][1] - sigma*rho[0]*S[n][0] + omega[n][1]*V1[n][0]
        dSdt[n] = - sum(p2[n])*alpha[n][1]*A[n][1]*S[n][1]*infect[n] + omega[n][2]*S[n][2] - omega[n][1]*S[n][1] - sigma*rho[0]*S[n][1] + Gamma[n][1]*I[n][1] + omega[n][2]*V1[n][1]
        dSdt[n] = - (p3[n][1]+p3[n][2])*alpha[n][2]*A[n][2]*S[n][2]*infect[n] + omega[n][3]*S[n][3] - omega[n][2]*S[n][2] - sigma*rho[0]*S[n][2] + Gamma[n][2]*I[n][2] + omega[n][3]*(V1[n][2]+V1[n][3]+sum(V2[n]))
        dSdt[n] = - omega[n][3]*S[n][3] - sigma*rho[0]*S[n][3] + Gamma[n][3]*I[n][3]
        
        # Vaccinated compartments
        dV1dt[n] = sigma*rho[0]*S[n][0] - sigma*rho[0]*V1[n][0] - sum(p2[n])*alpha[n][1]*A[n][1]*V1[n][0]*infect[n] - omega[n][1]*V1[n][0]
        dV1dt[n] = sigma*rho[0]*S[n][1] - sigma*rho[0]*V1[n][1] - (p3[n][1]+p3[n][2])*alpha[n][2]*A[n][2]*V1[n][1]*infect[n] - omega[n][2]*V1[n][1]
        dV1dt[n] = sigma*rho[0]*S[n][2] - sigma*rho[0]*V1[n][2] - omega[n][3]*V1[n][2]
        dV1dt[n] = sigma*rho[0]*S[n][3] - sigma*rho[0]*V1[n][3] - omega[n][3]*V1[n][3]

        dV2dt[n] = sigma*rho[0]*V1[n][0] - omega[n][3]*V2[n][0]
        dV2dt[n] = sigma*rho[0]*V1[n][1] - omega[n][3]*V2[n][1]
        dV2dt[n] = sigma*rho[0]*V1[n][2] - omega[n][3]*V2[n][2]
        dV2dt[n] = sigma*rho[0]*V1[n][3] - omega[n][3]*V2[n][3]

        # Exposed compartments
        dE2dt[n] = p1[n][0]*alpha[n][0]*A[n][0]*S[n][0]*infect[n] + p2[n][0]*alpha[n][1]*A[n][1]*S[n][1]*infect[n] + p2[n][0]*alpha[n][1]*A[n][1]*V1[n][0]*infect[n] - kappa[n][1]*E2[n][1]
        dE2dt[n] = p1[n][1]*alpha[n][0]*A[n][0]*S[n][0]*infect[n] + p2[n][1]*alpha[n][1]*A[n][1]*S[n][1]*infect[n] + p3[n][1]*alpha[n][2]*A[n][2]*S[n][2]*infect[n] + p2[n][1]*alpha[n][1]*A[n][1]*V1[n][0]*infect[n] + p3[n][1]*alpha[n][2]*A[n][2]*V1[n][1]*infect[n] - kappa[n][2]*E2[n][2]
        dE2dt[n] = p1[n][2]*alpha[n][0]*A[n][0]*S[n][0]*infect[n] + p2[n][2]*alpha[n][1]*A[n][1]*S[n][1]*infect[n] + p3[n][2]*alpha[n][2]*A[n][2]*S[n][2]*infect[n] + p2[n][2]*alpha[n][1]*A[n][1]*V1[n][0]*infect[n] + p3[n][2]*alpha[n][2]*A[n][2]*V1[n][1]*infect[n] - kappa[n][3]*E2[n][3]

        dE3dt[n] = kappa[n][1]*E2[n][1] - kappa[n][1]*E3[n][1]
        dE3dt[n] = kappa[n][2]*E2[n][2] - kappa[n][2]*E3[n][2]
        dE3dt[n] = kappa[n][3]*E2[n][3] - kappa[n][3]*E3[n][3]

        dE4dt[n] = kappa[n][1]*E3[n][1] - kappa[n][1]*E4[n][1]
        dE4dt[n] = kappa[n][2]*E3[n][2] - kappa[n][2]*E4[n][2]
        dE4dt[n] = kappa[n][3]*E3[n][3] - kappa[n][3]*E4[n][3]            

        # Infected compartments
        dIdt[n] = kappa[n][1]*E4[n][1] - delta[n][1]*I[n][1] - Gamma[n][1]*I[n][1]
        dIdt[n] = kappa[n][2]*E4[n][2] - delta[n][2]*I[n][2] - Gamma[n][2]*I[n][2]
        dIdt[n] = kappa[n][3]*E4[n][3] - delta[n][3]*I[n][3] - Gamma[n][3]*I[n][3] 

    return dSdt, dE2dt, dE3dt, dE4dt, dV1dt, dV2dt, dIdt


# Initial conditions
S0 = create_list(0, N, 4)
for i in range(len(S)):
    S0[i][0]=Pop[i]
    if i in [4,5,6,7,8,9]:
        S0[i][0]=Pop[i]-1
E20 = create_list(0, N, 4)
E30 = create_list(0, N, 4)
E40 = create_list(0, N, 4)
I0 = create_list(0, N, 4)
for i in range(len(I0)):
    if i in [4,5,6,7,8,9]:
        I0[i]=[0, 10/6, 1/6, 0]
V10 = create_list(0, N, 4)
V20 = create_list(0, N, 4)

Nt0 = np.matrix(S0) + np.matrix(E20) + np.matrix(E30) + np.matrix(E40) + np.matrix(I0) + np.matrix(V10) + np.matrix(V20)
Nt0 = Nt0.tolist() 
y0 = np.array(S0), np.array(E20), np.array(E30), np.array(E40), np.array(V10), np.array(V20), np.array(I0)



# MAIN
step = 0 # number used to track how many time we changed contact matrices and k-value
final = [] # stock odeint output at each run
for i in range(0, 400):
    if (i == timeBreaks[step]):     # if day i is equals to a day where there is a change in NPI
        k = get_k_value(0, step)    # update k-value
        con, A, c = contactModifiersComputation(k)  # update contact matrices
        Nt = y0[0] + y0[1] + y0[2] + y0[3] + y0[4] + y0[5] + y0[6]  # calculation of total population 
        Ntt = sum(Nt)
        Ntt = sum(Ntt.tolist())     # total population
        # Infectivity of S class. Has to be adapt to be class-dependant
        _Xm = np.multiply(np.multiply(np.matrix(beta)+np.matrix(beta), nu(step)), np.matrix(y0[6])).tolist()
        Xm = [sum(x) for x in _Xm]
        Ym = np.divide(Xm, Ntt)
        infect = np.matmul(c,Ym.T).tolist()
        sigma = sigma_calculation(step)     # update vaccination rate
        step += 1
    t = np.linspace(0,2,2) # run the solver for two days
    ret = odeintw(model, y0, t, args=(p1, p2, p3, alpha, kappa, Gamma, rho, omega, delta, A, infect, sigma))
    if(i == 0): # for the first day, we retrive it
        final.append(ret[0])
    final.append(ret[1]) # we get the second day since the first is the same as the second in the last iteration
    y0 = ret[1] # y0 is updated to the current day



# Plotting
S4 = generate_age_output(final, 5, 1, 3)    # Compartment I4 (#6 and #4 respectively) of age group #1
fig, (ax1) = plt.subplots(1)
ax1.plot(S4, '-c', label='I4 - 10:19 ans')
plt.show()
