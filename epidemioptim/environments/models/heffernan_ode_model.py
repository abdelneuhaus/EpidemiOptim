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
        Vaccination efficacy for the first dose.
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

    # Susceptible compartments
    dS1dt = - sum(p1)*alpha[0]*A[0]*S1*infect + omega[1]*S2 - sigma*rho*S1 + omega[1]*V11
    dS2dt = - sum(p2)*alpha[1]*A[1]*S2*infect + omega[2]*S3 - omega[1]*S2 - sigma*rho*S2 + gamma[1]*I2 + omega[2]*V12
    dS3dt = - (p3[1]+p3[2])*alpha[2]*A[2]*S3*infect + omega[3]*S4 - omega[2]*S3 - sigma*rho*S3 + gamma[2]*I3 + omega[3]*(V13+V14+V21+V22+V23+V24)
    dS4dt = - omega[3]*S4 - sigma*rho*S4 + gamma[3]*I4
    
    # Vaccinated compartments
    dV11dt = sigma*rho*S1 - sigma*rho*V11 - sum(p2)*alpha[1]*A[1]*V11*infect - omega[1]*V11
    dV12dt = sigma*rho*S2 - sigma*rho*V12 - (p3[1]+p3[2])*alpha[2]*A[2]*V12*infect - omega[2]*V12
    dV13dt = sigma*rho*S3 - sigma*rho*V13 - omega[3]*V13
    dV14dt = sigma*rho*S4 - sigma*rho*V14 - omega[3]*V14

    dV21dt = sigma*rho*V11 - omega[3]*V21
    dV22dt = sigma*rho*V12 - omega[3]*V22
    dV23dt = sigma*rho*V13 - omega[3]*V23
    dV24dt = sigma*rho*V14 - omega[3]*V24

    # Exposed compartments
    dE21dt = p1[0]*alpha[0]*A[0]*S1*infect + p2[0]*alpha[1]*A[1]*S2*infect + p2[0]*alpha[1]*A[1]*V11*infect - kappa[1]*E21
    dE22dt = p1[1]*alpha[0]*A[0]*S1*infect + p2[1]*alpha[1]*A[1]*S2*infect + p3[1]*alpha[2]*A[2]*S3*infect + p2[1]*alpha[1]*A[1]*V11*infect + p3[1]*alpha[2]*A[2]*V12*infect - kappa[2]*E22
    dE23dt = p1[2]*alpha[0]*A[0]*S1*infect + p2[2]*alpha[1]*A[1]*S2*infect + p3[2]*alpha[2]*A[2]*S3*infect + p2[2]*alpha[1]*A[1]*V11*infect + p3[2]*alpha[2]*A[2]*V12*infect - kappa[3]*E23

    dE31dt = kappa[1]*E21 - kappa[1]*E31
    dE32dt = kappa[2]*E22 - kappa[2]*E32
    dE33dt = kappa[3]*E23 - kappa[3]*E33

    dE41dt = kappa[1]*E31 - kappa[1]*E41
    dE42dt = kappa[2]*E32 - kappa[2]*E42
    dE43dt = kappa[3]*E33 - kappa[3]*E43            

    # Infected compartments
    dI2dt = kappa[1]*E41 - delta[1]*I2 - gamma[1]*I2
    dI3dt = kappa[2]*E42 - delta[2]*I3 - gamma[2]*I3
    dI4dt = kappa[3]*E43 - delta[3]*I4 - gamma[3]*I4 

    dydt = [dS1dt, dS2dt, dS3dt, dS4dt, dE21dt, dE22dt, dE23dt, dE31dt, dE32dt, dE33dt, 
    dE41dt, dE42dt, dE43dt, dV11dt, dV12dt, dV13dt, dV14dt, dV21dt, dV22dt, dV23dt, dV24dt, dI2dt, dI3dt, dI4dt]
    return dydt

    # S, E2, E3, E4, I, V1, V2 = y

    # # For each age groups :
    # for n in range(0, 16):

    #     # Susceptible compartments
    #     dSdt = - sum(p1)*alpha[0]*A[0]*S[0]*infect + omega[1]*S[1] - sigma*rho*S[0] + omega[1]*V1[0]
    #     dSdt = - sum(p2)*alpha[1]*A[1]*S[1]*infect + omega[2]*S[2] - omega[1]*S[1] - sigma*rho*S[1] + gamma[1]*I[1] + omega[2]*V1[1]
    #     dSdt = - (p3[1]+p3[2])*alpha[2]*A[2]*S[2]*infect + omega[3]*S[3] - omega[2]*S[2] - sigma*rho*S[2] + gamma[2]*I[2] + omega[3]*(V1[2]+V1[3]+sum(V2))
    #     dSdt = - omega[3]*S[3] - sigma*rho*S[3] + gamma[3]*I[3]
        
    #     # Vaccinated compartments
    #     dV1dt = sigma*rho*S[0] - sigma*rho*V1[0] - sum(p2)*alpha[1]*A[1]*V1[0]*infect - omega[1]*V1[0]
    #     dV1dt = sigma*rho*S[1] - sigma*rho*V1[1] - (p3[1]+p3[2])*alpha[2]*A[2]*V1[1]*infect - omega[2]*V1[1]
    #     dV1dt = sigma*rho*S[2] - sigma*rho*V1[2] - omega[3]*V1[2]
    #     dV1dt = sigma*rho*S[3] - sigma*rho*V1[3] - omega[3]*V1[3]

    #     dV2dt = sigma*rho*V1[0] - omega[3]*V2[0]
    #     dV2dt = sigma*rho*V1[1] - omega[3]*V2[1]
    #     dV2dt = sigma*rho*V1[2] - omega[3]*V2[2]
    #     dV2dt = sigma*rho*V1[3] - omega[3]*V2[3]

    #     # Exposed compartments
    #     dE2dt = p1[0]*alpha[0]*A[0]*S[0]*infect + p2[0]*alpha[1]*A[1]*S[1]*infect + p2[0]*alpha[1]*A[1]*V1[0]*infect - kappa[1]*E2[1]
    #     dE2dt = p1[1]*alpha[0]*A[0]*S[0]*infect + p2[1]*alpha[1]*A[1]*S[1]*infect + p3[1]*alpha[2]*A[2]*S[2]*infect + p2[1]*alpha[1]*A[1]*V1[0]*infect + p3[1]*alpha[2]*A[2]*V1[1]*infect - kappa[2]*E2[2]
    #     dE2dt = p1[2]*alpha[0]*A[0]*S[0]*infect + p2[2]*alpha[1]*A[1]*S[1]*infect + p3[2]*alpha[2]*A[2]*S[2]*infect + p2[2]*alpha[1]*A[1]*V1[0]*infect + p3[2]*alpha[2]*A[2]*V1[1]*infect - kappa[3]*E2[3]

    #     dE3dt = kappa[1]*E2[1] - kappa[1]*E3[1]
    #     dE3dt = kappa[2]*E2[2] - kappa[2]*E3[2]
    #     dE3dt = kappa[3]*E2[3] - kappa[3]*E3[3]

    #     dE4dt = kappa[1]*E3[1] - kappa[1]*E4[1]
    #     dE4dt = kappa[2]*E3[2] - kappa[2]*E4[2]
    #     dE4dt = kappa[3]*E3[3] - kappa[3]*E4[3]            

    #     # Infected compartments
    #     dIdt = kappa[1]*E4[1] - delta[1]*I[1] - gamma[1]*I[1]
    #     dIdt = kappa[2]*E4[2] - delta[2]*I[2] - gamma[2]*I[2]
    #     dIdt = kappa[3]*E4[3] - delta[3]*I[3] - gamma[3]*I[3] 

    # return dSdt, dE2dt, dE3dt, dE4dt, dV1dt, dV2dt, dIdt


class HeffernanOdeModel(BaseModel):
    def __init__(self,
                stochastic=False,
                range_delay=None
                ):
        self._age_groups = [['0-4'], ['5-9'], ['10-14'], ['15-19'], ['20-24'], ['25-29'], ['30-34'], ['35-39'], ['40-44'], ['45-49'], ['50-54'], ['55-59'], ['60-64'], ['65-69'],  ['70-74'], ['75+']]
        self.common_parameters = {"alpha": [1, 2/3, 1/3, 0], "beta": [0.08, 0.04, 0.08, 0.008], "gamma": [0, 0.2, 0.1, 1/15], "delta": [0, 0, 0, 0.0001], "omega": [0, 1/365, 1/365, 1/365], "kappa": [0, 1/1.5, 1/1.5, 1/1.5], "rho": 0.8, "N": 16}
        self._pop_size = pd.read_excel(PATH_TO_DATA, sheet_name='population', skiprows=3, usecols=(2,2))
        self.pop_size = None

print( pd.read_excel(PATH_TO_DATA, sheet_name='population', skiprows=3, usecols=(2,2)))

