from odeintw import odeintw


from epidemioptim.environments.models.base_model import BaseModel
from epidemioptim.utils import *

PATH_TO_DATA = get_repo_path() + '/data/jane_model_data/ScenarioPlanFranceOne.xlsx'
PATH_TO_HOME_MATRIX = get_repo_path() + '/data/jane_model_data/contactHome.txt'
PATH_TO_SCHOOL_MATRIX = get_repo_path() + '/data/jane_model_data/contactSchool.txt'
PATH_TO_WORK_MATRIX = get_repo_path() + '/data/jane_model_data/contactWork.txt'
PATH_TO_OTHER_MATRIX = get_repo_path() + '/data/jane_model_data/contactOtherPlaces.txt'
PATH_TO_COMORBIDITY_MATRIX = get_repo_path() + '/data/jane_model_data/coMorbidity.txt'


# ODE model
def model(y, t, p1, p2, p3, alpha, kappa, Gamma, rho, omega, delta, A, infect, sigma):
    S, E2, E3, E4, I, V1, V2 = y

    # For each age groups :
    for n in range(0, 16):

        # Susceptible compartments
        dS1dt = - sum(p1[n])*alpha[n][0]*A[n][0]*S[n][0]*infect[n] + omega[n][1]*S[n][1] - sigma*rho[0]*S[n][0] + omega[n][1]*V1[n][0]
        dS2dt = - sum(p2[n])*alpha[n][1]*A[n][1]*S[n][1]*infect[n] + omega[n][2]*S[n][2] - omega[n][1]*S[n][1] - sigma*rho[0]*S[n][1] + Gamma[n][1]*I[n][1] + omega[n][2]*V1[n][1]
        dS3dt = - (p3[n][1]+p3[n][2])*alpha[n][2]*A[n][2]*S[n][2]*infect[n] + omega[n][3]*S[n][3] - omega[n][2]*S[n][2] - sigma*rho[0]*S[n][2] + Gamma[n][2]*I[n][2] + omega[n][3]*(V1[n][2]+V1[n][3]+sum(V2[n]))
        dS4dt = - omega[n][3]*S[n][3] - sigma*rho[0]*S[n][3] + Gamma[n][3]*I[n][3]
        
        # Vaccinated compartments
        dV11dt = sigma*rho[0]*S[n][0] - sigma*rho[0]*V1[n][0] - sum(p2[n])*alpha[n][1]*A[n][1]*V1[n][0]*infect[n] - omega[n][1]*V1[n][0]
        dV12dt = sigma*rho[0]*S[n][1] - sigma*rho[0]*V1[n][1] - (p3[n][1]+p3[n][2])*alpha[n][2]*A[n][2]*V1[n][1]*infect[n] - omega[n][2]*V1[n][1]
        dV13dt = sigma*rho[0]*S[n][2] - sigma*rho[0]*V1[n][2] - omega[n][3]*V1[n][2]
        dV14dt = sigma*rho[0]*S[n][3] - sigma*rho[0]*V1[n][3] - omega[n][3]*V1[n][3]

        dV21dt = sigma*rho[0]*V1[n][0] - omega[n][3]*V2[n][0]
        dV22dt = sigma*rho[0]*V1[n][1] - omega[n][3]*V2[n][1]
        dV23dt = sigma*rho[0]*V1[n][2] - omega[n][3]*V2[n][2]
        dV24dt = sigma*rho[0]*V1[n][3] - omega[n][3]*V2[n][3]

        # Exposed compartments
        dE21dt = p1[n][0]*alpha[n][0]*A[n][0]*S[n][0]*infect[n] + p2[n][0]*alpha[n][1]*A[n][1]*S[n][1]*infect[n] + p2[n][0]*alpha[n][1]*A[n][1]*V1[n][0]*infect[n] - kappa[n][1]*E2[n][1]
        dE22dt = p1[n][1]*alpha[n][0]*A[n][0]*S[n][0]*infect[n] + p2[n][1]*alpha[n][1]*A[n][1]*S[n][1]*infect[n] + p3[n][1]*alpha[n][2]*A[n][2]*S[n][2]*infect[n] + p2[n][1]*alpha[n][1]*A[n][1]*V1[n][0]*infect[n] + p3[n][1]*alpha[n][2]*A[n][2]*V1[n][1]*infect[n] - kappa[n][2]*E2[n][2]
        dE23dt = p1[n][2]*alpha[n][0]*A[n][0]*S[n][0]*infect[n] + p2[n][2]*alpha[n][1]*A[n][1]*S[n][1]*infect[n] + p3[n][2]*alpha[n][2]*A[n][2]*S[n][2]*infect[n] + p2[n][2]*alpha[n][1]*A[n][1]*V1[n][0]*infect[n] + p3[n][2]*alpha[n][2]*A[n][2]*V1[n][1]*infect[n] - kappa[n][3]*E2[n][3]

        dE31dt = kappa[n][1]*E2[n][1] - kappa[n][1]*E3[n][1]
        dE32dt = kappa[n][2]*E2[n][2] - kappa[n][2]*E3[n][2]
        dE33dt = kappa[n][3]*E2[n][3] - kappa[n][3]*E3[n][3]

        dE41dt = kappa[n][1]*E3[n][1] - kappa[n][1]*E4[n][1]
        dE42dt = kappa[n][2]*E3[n][2] - kappa[n][2]*E4[n][2]
        dE43dt = kappa[n][3]*E3[n][3] - kappa[n][3]*E4[n][3]            

        # Infected compartments
        dI2dt = kappa[n][1]*E4[n][1] - delta[n][1]*I[n][1] - Gamma[n][1]*I[n][1]
        dI3dt = kappa[n][2]*E4[n][2] - delta[n][2]*I[n][2] - Gamma[n][2]*I[n][2]
        dI4dt = kappa[n][3]*E4[n][3] - delta[n][3]*I[n][3] - Gamma[n][3]*I[n][3] 

    dydt = [dS1dt, dS2dt, dS3dt, dS4dt, dE21dt, dE22dt, dE23dt, dE31dt, dE32dt, dE33dt, 
    dE41dt, dE42dt, dE43dt, dV11dt, dV12dt, dV13dt, dV14dt, dV21dt, dV22dt, dV23dt, dV24dt, dI2dt, dI3dt, dI4dt]
    return dydt