from numpy import intersect1d
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
def vaccination_model(y: tuple,
          t: tuple,
          p1: tuple, 
          p2: tuple, 
          p3: tuple, 
          alpha: tuple,
          beta: tuple, 
          kappa: tuple, 
          gamma: tuple, 
          rho: float, 
          omega: tuple, 
          delta: tuple, 
          A: tuple, 
          c: tuple, 
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
    c: tuple
       Mixing matrix between individuals in age group a and age groupe n, modified given mitigation, strategy, PPE, 
       social distancing, hand washing compliance (k-value)
    sigma: float
       Vaccination rate.

    Returns
    -------
    tuple
        Next states.
    """

    S1, S2, S3, S4, E21, E22, E23, E31, E32, E33, E41, E42, E43, V11, V12, V13, V14, V21, V22, V23, V24, I2, I3, I4 = y
    T = S1 + S2 + S3 + S4 + E21 + E22 + E23 + E31 + E32 + E33 + E41 + E42 + E43 + V11 + V12 + V13 + V14 + V21 + V22 + V23 + V24 + I2 + I3 + I4
    infect = sum(c)*((beta[1]+beta[2]+beta[3])*(I2+I3+I4)/T)

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



class HeffernanOdeModel(BaseModel):
    def __init__(self,
                age_group = '0-4',
                stochastic=False,
                range_delay=None
                ):
        self.p1 = get_text_file_data(PATH_TO_COMORBIDITY_MATRIX)
        self.p2 = get_text_file_data(PATH_TO_COMORBIDITY_MATRIX)
        self.p3 = [[0] + sub[1:] for sub in self.p1]
        self.kval = get_kvalue(PATH_TO_DATA)
        self.work = get_text_file_data(PATH_TO_WORK_MATRIX)
        self.other = get_text_file_data(PATH_TO_OTHER_MATRIX)
        self.home = get_text_file_data(PATH_TO_HOME_MATRIX)
        self.school = get_text_file_data(PATH_TO_SCHOOL_MATRIX)
        self.perturbations_matrices = get_perturbations_matrices(PATH_TO_DATA)
        self.contact_modifiers = get_contact_modifiers(PATH_TO_DATA)
        self.vaccination_coverage = get_coverage(PATH_TO_DATA)
        self.active_vaccination = vaccination_active(PATH_TO_DATA)
        self.transition_matrices = transition_matrices(self.pop_size, self.H, self.S, self.W, self.O)
        self._age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',  '70-74', '75+']
        self._pop_size = pd.read_excel(PATH_TO_DATA, sheet_name='population', skiprows=3, usecols=(2,2))
        self.pop_size = dict(zip(self._age_groups, (self._pop_size['Unnamed: 2'])))
        self.parameters = ['alpha', 'beta', 'gamma', 'delta', 'omega', 'kappa', 'rho', 'sigma', 'p1', 'p2', 'p3', 'A', 'infect']
        self.step = [0, 71, 73, 76, 153, 173, 185, 201, 239, 244, 290, 295, 303, 305, 349, 353, 369, 370, 377, 381, 384, 391, 398, 402, 
                     404, 405, 409, 412, 418, 419, 425, 426, 431, 433, 440, 447, 454, 459, 461, 465, 468, 472 , 475, 481, 482, 488, 
                     489, 494, 496, 497, 501, 503, 510, 517, 524, 531, 552, 592, 609, 731]
        assert age_group in self._age_groups, 'age group should be one of ' + str(self._regions)

        self.age_group = age_group
        self.stochastic = stochastic
        self._all_internal_params_distribs = dict()
        self._all_initial_state_distribs = dict()

        # Initialize distributions of parameters and initial conditions for all regions
        self.define_params_and_initial_state_distributions()

        # Sample initial conditions and initial model parameters
        internal_params_labels = ['alpha', 'beta', 'gamma', 'delta', 'omega', 'kappa', 'rho', 'sigma', 'p1', 'p2', 'p3', 'A', 'infect']

        # Define ODE model
        self.internal_model = vaccination_model

        super().__init__(internal_states_labels=['S1', 'S2', 'S3', 'S4', 'E21', 'E22', 'E23', 'E31', 'E32', 'E33', 'E41', 'E42', 'E43',
                                                 'V11', 'V12', 'V13', 'V14', 'V21', 'V22', 'V23', 'V24', 'I2', 'I3', 'I4'],
                         internal_params_labels=internal_params_labels,
                         stochastic=stochastic,
                         range_delay=range_delay)


    def define_params_and_initial_state_distributions(self):
        """
        Extract and define distributions of parameters for all age groups
        """

        grp = 0
        for i in self._age_groups[i]:
            self._all_internal_params_distribs[i] = dict(alpha=[1, 2/3, 1/3, 0],
                                                         beta=[0.08, 0.04, 0.08, 0.008],
                                                         gamma=[0, 0.2, 0.1, 1/15],
                                                         delta=[0, 0, 0, 0.0001],
                                                         omega=[0, 1/365, 1/365, 1/365],
                                                         kappa=[0, 1/1.5, 1/1.5, 1/1.5],
                                                         rho=0.8,
                                                         sigma=1e-20,
                                                         p1=self.p1[grp],
                                                         p2=self.p2[grp],
                                                         p3=self.p3[grp],
                                                         A=calculate_A_and_c(0, 1, self.contact_modifiers, self.perturbations_matrices, self.transition_matrices, 16)[0],
                                                         c=calculate_A_and_c(0, 1, self.contact_modifiers, self.perturbations_matrices, self.transition_matrices, 16)[1],
                                                         sigma=sigma_calculation(0, 0, self.vaccination_coverage)
                                                        )
                                                        
            self._all_initial_state_distribs[i] = dict(S20=DiracDist(params=0, stochastic=self.stochastic),
                                                       S30=DiracDist(params=0, stochastic=self.stochastic),
                                                       S40=DiracDist(params=0, stochastic=self.stochastic),
                                                       E210=DiracDist(params=0, stochastic=self.stochastic),
                                                       E220=DiracDist(params=0, stochastic=self.stochastic),
                                                       E230=DiracDist(params=0, stochastic=self.stochastic),
                                                       E310=DiracDist(params=0, stochastic=self.stochastic),
                                                       E320=DiracDist(params=0, stochastic=self.stochastic),
                                                       E330=DiracDist(params=0, stochastic=self.stochastic),
                                                       E410=DiracDist(params=0, stochastic=self.stochastic),
                                                       E420=DiracDist(params=0, stochastic=self.stochastic),
                                                       E430=DiracDist(params=0, stochastic=self.stochastic),
                                                       V110=DiracDist(params=0, stochastic=self.stochastic),
                                                       V120=DiracDist(params=0, stochastic=self.stochastic),
                                                       V130=DiracDist(params=0, stochastic=self.stochastic),
                                                       V140=DiracDist(params=0, stochastic=self.stochastic),
                                                       V210=DiracDist(params=0, stochastic=self.stochastic),
                                                       V220=DiracDist(params=0, stochastic=self.stochastic),
                                                       V230=DiracDist(params=0, stochastic=self.stochastic),
                                                       V240=DiracDist(params=0, stochastic=self.stochastic),
                                                       I20=DiracDist(params=10/6, stochastic=self.stochastic),
                                                       I30=DiracDist(params=1/6, stochastic=self.stochastic),
                                                       I40=DiracDist(params=0, stochastic=self.stochastic)
                                                       )
            grp += 1                                           


parameters = ['alpha', 'beta', 'gamma', 'delta', 'omega', 'kappa', 'rho', 'sigma', 'p1', 'p2', 'p3', 'A', 'infect']
_age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',  '70-74', '75+']
_pop_size = pd.read_excel(PATH_TO_DATA, sheet_name='population', skiprows=3, usecols=(2,2))
label2ind = dict(zip(list(parameters), np.arange(len(_age_groups))))
_all_internal_params_distribs = dict()
for i in _age_groups:
    _all_internal_params_distribs[i] = [0]
    
pertubations_matrices = get_perturbations_matrices(PATH_TO_DATA)
pop_size = dict(zip(_age_groups, (_pop_size['Unnamed: 2'])))

contact_modifiers = get_contact_modifiers(PATH_TO_DATA)
perturbations_matrices = get_perturbations_matrices(PATH_TO_DATA)
W = get_text_file_data(PATH_TO_WORK_MATRIX) # work
O = get_text_file_data(PATH_TO_OTHER_MATRIX) # other places
H = get_text_file_data(PATH_TO_HOME_MATRIX) # home
S = get_text_file_data(PATH_TO_SCHOOL_MATRIX)
transition_matrices = transition_matrices(pop_size, H, S, W, O)

A = calculate_A_and_c(0, 1, contact_modifiers, perturbations_matrices, transition_matrices, 16)[0]
print(A)