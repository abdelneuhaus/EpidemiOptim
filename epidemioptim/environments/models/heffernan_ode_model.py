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
def model(y: tuple,
          t: tuple,
          p1: tuple, 
          p2: tuple, 
          p3: tuple, 
          alpha: tuple, 
          kappa: tuple, 
          gamma: tuple, 
          rho: float, 
          omega: tuple, 
          delta: tuple, 
          A: tuple, 
          infect: tuple, 
          sigma: float):
    """
    Parameters
    ----------
    y: tuple
       y = [S1, S2, S3, S4, E21, E22, E23, E31, E32, E33, E41, E42, E43, V11, V12, V13, V14, V21, V22, V23, V24, I2, I3, I4]
       Si: # susceptible individuals with i level of infectiosity
       E2i: # individuals in mild latent state
       E3i: # individuals in moderate latent state
       E4i: # individuals in severe latent state
       Ii: # symptomatic infected individuals with i level of infectiosity
       V1i: # vaccinated people with one dose, i being the immunity level
       V2i: # vaccinated people with two doses, i being the immunity level
    t: int
       Timestep.
    p1: tuple
        Probability to go to mild class for an age group.
    p2: tuple
        Probability to go to moderate class for an age group.
    p3: tuple
        Probability to go to severe class for an age group.        
    alpha: tuple
        Susceptibilty of individuals from Sin (i immunity status, n age group).
    kappa: tuple
        Rates of progress through the pre-infectious period of infection.
    gamma: tuple
        Recovery rate of infected individuals from Ijm (j immunity status, m age group).
    rho: float
        Vaccination efficacy.
    omega: tuple
        Waning rate of immunity of individuals from Sin (i immunity status, n age group).
    delta: tuple
       Disease-induced mortality rate of infected individuals from Ijm (j immunity status, m age group).
    A: tuple
       Per capita activity counts of individuals in age group n
    infect: tuple
       Force of infection. Product of 
    sigma: float
       Vaccination rate.

    Returns
    -------
    tuple
        Next states.
    """
    S1, S2, S3, S4, E21, E22, E23, E31, E32, E33, E41, E42, E43, V11, V12, V13, V14, V21, V22, V23, V24, I2, I3, I4 = y

    # For each age groups :
    for n in range(0, 16):

        # Susceptible compartments
        dS1dt = - sum(p1[n])*alpha[n][0]*A[n][0]*S1[n]*infect[n] + omega[n][1]*S2[n] - sigma*rho[0]*S1[n] + omega[n][1]*V11[n]
        dS2dt = - sum(p2[n])*alpha[n][1]*A[n][1]*S2[n]*infect[n] + omega[n][2]*S3[n] - omega[n][1]*S2[n] - sigma*rho[0]*S2[n] + gamma[n][1]*I2[n] + omega[n][2]*V12[n]
        dS3dt = - (p3[n][1]+p3[n][2])*alpha[n][2]*A[n][2]*S3[n]*infect[n] + omega[n][3]*S4[n] - omega[n][2]*S3[n] - sigma*rho[0]*S3[n] + gamma[n][2]*I3[n] + omega[n][3]*(V13[n]+V14[n]+V21[n]+V22[n]+V23[n]+V24[n])
        dS4dt = - omega[n][3]*S4[n] - sigma*rho[0]*S4[n] + gamma[n][3]*I4[n]
        
        # Vaccinated compartments
        dV11dt = sigma*rho[0]*S1[n] - sigma*rho[0]*V11[n] - sum(p2[n])*alpha[n][1]*A[n][1]*V11[n]*infect[n] - omega[n][1]*V11[n]
        dV12dt = sigma*rho[0]*S2[n] - sigma*rho[0]*V12[n] - (p3[n][1]+p3[n][2])*alpha[n][2]*A[n][2]*V12[n]*infect[n] - omega[n][2]*V12[n]
        dV13dt = sigma*rho[0]*S3[n] - sigma*rho[0]*V13[n] - omega[n][3]*V13[n]
        dV14dt = sigma*rho[0]*S4[n] - sigma*rho[0]*V14[n] - omega[n][3]*V14[n]

        dV21dt = sigma*rho[0]*V11[n] - omega[n][3]*V21[n]
        dV22dt = sigma*rho[0]*V12[n] - omega[n][3]*V22[n]
        dV23dt = sigma*rho[0]*V13[n] - omega[n][3]*V23[n]
        dV24dt = sigma*rho[0]*V14[n] - omega[n][3]*V24[n]

        # Exposed compartments
        dE21dt = p1[n][0]*alpha[n][0]*A[n][0]*S1[n]*infect[n] + p2[n][0]*alpha[n][1]*A[n][1]*S2[n]*infect[n] + p2[n][0]*alpha[n][1]*A[n][1]*V11[n]*infect[n] - kappa[n][1]*E21[n]
        dE22dt = p1[n][1]*alpha[n][0]*A[n][0]*S1[n]*infect[n] + p2[n][1]*alpha[n][1]*A[n][1]*S2[n]*infect[n] + p3[n][1]*alpha[n][2]*A[n][2]*S3[n]*infect[n] + p2[n][1]*alpha[n][1]*A[n][1]*V11[n]*infect[n] + p3[n][1]*alpha[n][2]*A[n][2]*V12[n]*infect[n] - kappa[n][2]*E22[n]
        dE23dt = p1[n][2]*alpha[n][0]*A[n][0]*S1[n]*infect[n] + p2[n][2]*alpha[n][1]*A[n][1]*S2[n]*infect[n] + p3[n][2]*alpha[n][2]*A[n][2]*S3[n]*infect[n] + p2[n][2]*alpha[n][1]*A[n][1]*V11[n]*infect[n] + p3[n][2]*alpha[n][2]*A[n][2]*V12[n]*infect[n] - kappa[n][3]*E23[n]

        dE31dt = kappa[n][1]*E21[n] - kappa[n][1]*E31[n]
        dE32dt = kappa[n][2]*E22[n] - kappa[n][2]*E32[n]
        dE33dt = kappa[n][3]*E23[n] - kappa[n][3]*E33[n]

        dE41dt = kappa[n][1]*E31[n] - kappa[n][1]*E41[n]
        dE42dt = kappa[n][2]*E32[n] - kappa[n][2]*E42[n]
        dE43dt = kappa[n][3]*E33[n] - kappa[n][3]*E43[n]            

        # Infected compartments
        dI2dt = kappa[n][1]*E41[n] - delta[n][1]*I2[n] - gamma[n][1]*I2[n]
        dI3dt = kappa[n][2]*E42[n] - delta[n][2]*I3[n] - gamma[n][2]*I3[n]
        dI4dt = kappa[n][3]*E43[n] - delta[n][3]*I4[n] - gamma[n][3]*I4[n] 

    dydt = [dS1dt, dS2dt, dS3dt, dS4dt, dE21dt, dE22dt, dE23dt, dE31dt, dE32dt, dE33dt, 
    dE41dt, dE42dt, dE43dt, dV11dt, dV12dt, dV13dt, dV14dt, dV21dt, dV22dt, dV23dt, dV24dt, dI2dt, dI3dt, dI4dt]
    return dydt