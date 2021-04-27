import numpy as np
from numpy.core.fromnumeric import cumsum
from numpy.lib.shape_base import dsplit
from scipy.integrate import odeint
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
                      t: float,
                      A: tuple,
                      alpha: tuple,
                      beta: tuple, 
                      c: tuple,
                      delta: tuple,
                      gamma: tuple,
                      kappa: tuple, 
                      omega: tuple,
                      p1: tuple, 
                      p2: tuple, 
                      p3: tuple,
                      rho: float, 
                      sigma: float
                      ):
    """
    Parameters
    ----------
    y: tuple
       y = [S1, S2, S3, S4, E21, E22, E23, E31, E32, E33, E41, E42, E43, V11, V12, V13, V14, V21, V22, V23, V24, I2, I3, I4]
       Si: # susceptible individuals with i level of infectivity
       E2i: # individuals in mild latent state
       E3i: # individuals in moderate latent state
       E4i: # individuals in severe latent state
       Ii: # symptomatic infected individuals with i level of infectivity
       V1i: # vaccinated people with one dose, i being the immunity level
       V2i: # vaccinated people with two doses, i being the immunity level
    t: int
       Timestep.
    data: dict
          Has parameters of all age groups. These parameters are:
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
    dS1dt, dS2dt, dS3dt, dS4dt = np.zeros((1,16)), np.zeros((1,16)), np.zeros((1,16)), np.zeros((1,16))
    dE21dt, dE22dt, dE23dt = np.zeros((1,16)), np.zeros((1,16)), np.zeros((1,16))
    dE31dt, dE32dt, dE33dt = np.zeros((1,16)), np.zeros((1,16)), np.zeros((1,16))
    dE41dt, dE42dt, dE43dt = np.zeros((1,16)), np.zeros((1,16)), np.zeros((1,16))
    dV11dt, dV12dt, dV31dt, dV41dt = np.zeros((1,16)), np.zeros((1,16)), np.zeros((1,16)), np.zeros((1,16))
    dV21dt, dV22dt, dV32dt, dV42dt = np.zeros((1,16)), np.zeros((1,16)), np.zeros((1,16)), np.zeros((1,16))
    dI2dt, dI3dt, dI4dt = np.zeros((1,16)), np.zeros((1,16)), np.zeros((1,16))

    for i in range(16):
        S1, S2, S3, S4 = y[i][0], y[i][1], y[i][2], y[i][3]
        E21, E22, E23 =  y[i][4], y[i][5], y[i][6]
        E31, E32, E33 =  y[i][7], y[i][8], y[i][9]
        E41, E42, E43 = y[i][10], y[i][11], y[i][12]
        V11, V21, V31, V41 = y[i][13], y[i][14], y[i][15], y[i][16]
        V12, V22, V32, V42 = y[i][17], y[i][18], y[i][19], y[i][20]
        I2, I3, I4 = y[i][21], y[i][22], y[i][23]
        T = S1 + S2 + S3 + S4 + E21 + E22 + E23 + E31 + E32 + E33 + E41 + E42 + E43 + V11 + V21 + V31 + V41 + V12 + V22 + V32 + V42 + I2 + I3 + I4
        infect = sum(c[i])*((beta[1]+beta[2]+beta[3])*(I2+I3+I4)/T)
        #print(infect)
        # Susceptible compartments
        dS1dt[0][i] = - sum(p1[i])*alpha[0]*A[i][0]*S1*infect + omega[1]*S2 - sigma*rho*S1 + omega[1]*V11
        print("S1", S1, "S2", S2)
        dS2dt[0][i] = - sum(p2[i])*alpha[1]*A[i][1]*S2*infect + omega[2]*S3 - omega[1]*S2 - sigma*rho*S2 + gamma[1]*I2 + omega[2]*V21
        dS3dt[0][i] = - (p3[i][1]+p3[i][2])*alpha[2]*A[i][2]*S3*infect + omega[3]*S4 - omega[2]*S3 - sigma*rho*S3 + gamma[2]*I3 + omega[3]*(V31+V41+V12+V22+V32+V42)
        dS4dt[0][i] = - omega[3]*S4 - sigma*rho*S4 + gamma[3]*I4

        # Vaccinated compartments
        dV11dt[0][i] = sigma*rho*S1 - sigma*rho*V11 - sum(p2[i])*alpha[1]*A[i][1]*V11*infect - omega[1]*V11
        dV21dt[0][i] = sigma*rho*S2 - sigma*rho*V21 - (p3[i][1]+p3[i][2])*alpha[2]*A[i][2]*V21*infect - omega[2]*V21
        dV31dt[0][i] = sigma*rho*S3 - sigma*rho*V31 - omega[3]*V31
        dV41dt[0][i] = sigma*rho*S4 - sigma*rho*V41 - omega[3]*V41

        dV12dt[0][i] = sigma*rho*V11 - omega[3]*V12
        dV22dt[0][i] = sigma*rho*V21 - omega[3]*V22
        dV32dt[0][i] = sigma*rho*V31 - omega[3]*V32
        dV42dt[0][i] = sigma*rho*V41 - omega[3]*V42

        # Exposed compartments
        # To I2
        dE21dt[0][i] = p1[i][0]*alpha[0]*A[i][0]*S1*infect + p2[i][0]*alpha[1]*A[i][1]*S2*infect + p2[i][0]*alpha[1]*A[i][1]*V11*infect - kappa[1]*E21
        dE22dt[0][i] = kappa[1]*E21 - kappa[2]*E22
        dE23dt[0][i] = kappa[2]*E22 - kappa[3]*E23

        # To I3
        dE31dt[0][i] = p1[i][1]*alpha[0]*A[i][0]*S1*infect + p2[i][1]*alpha[1]*A[i][1]*S2*infect + p3[i][1]*alpha[2]*A[i][2]*S3*infect + p2[i][1]*alpha[1]*A[i][1]*V11*infect + p3[i][1]*alpha[2]*A[i][2]*V21*infect - kappa[1]*E31
        dE32dt[0][i] = kappa[1]*E31 - kappa[2]*E32
        dE33dt[0][i] = kappa[2]*E32 - kappa[3]*E33

        # To I4
        dE41dt[0][i] = p1[i][2]*alpha[0]*A[i][0]*S1*infect + p2[i][2]*alpha[1]*A[i][1]*S2*infect + p3[i][2]*alpha[2]*A[i][2]*S3*infect + p2[i][2]*alpha[1]*A[i][1]*V11*infect + p3[i][2]*alpha[2]*A[i][2]*V21*infect - kappa[1]*E41
        dE42dt[0][i] = kappa[1]*E41 - kappa[2]*E42
        dE43dt[0][i] = kappa[2]*E42 - kappa[3]*E43            

        # Infected compartments
        dI2dt[0][i] = kappa[3]*E23 - delta[1]*I2 - gamma[1]*I2
        dI3dt[0][i] = kappa[3]*E33 - delta[2]*I3 - gamma[2]*I3
        dI4dt[0][i] = kappa[3]*E43 - delta[3]*I4 - gamma[3]*I4 

    dydt = dS1dt, dS2dt, dS3dt, dS4dt, dE21dt, dE22dt, dE23dt, dE31dt, dE32dt, dE33dt, dE41dt, dE42dt, dE43dt, dV11dt, dV12dt, dV31dt, dV41dt, dV21dt, dV22dt, dV32dt, dV42dt, dI2dt, dI3dt, dI4dt
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
        self.work = get_text_file_data(PATH_TO_WORK_MATRIX)
        self.other = get_text_file_data(PATH_TO_OTHER_MATRIX)
        self.home = get_text_file_data(PATH_TO_HOME_MATRIX)
        self.school = get_text_file_data(PATH_TO_SCHOOL_MATRIX)
        self.perturbations_matrices = get_perturbations_matrices(PATH_TO_DATA)
        self.contact_modifiers = get_contact_modifiers(PATH_TO_DATA)
        self.vaccination_coverage = get_coverage(PATH_TO_DATA)
        self.active_vaccination = vaccination_active(PATH_TO_DATA)
        self._age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',  '70-74', '75+']
        self._pop_size = pd.read_excel(PATH_TO_DATA, sheet_name='population', skiprows=3, usecols=(2,2))
        self.pop_size = dict(zip(self._age_groups, (self._pop_size['Unnamed: 2'])))
        self.transition_matrices = get_transition_matrices(self.pop_size, self.home, self.school, self.work, self.other)
        self.step_list = [0, 71, 73, 76, 153, 173, 185, 201, 239, 244, 290, 295, 303, 305, 349, 353, 369, 370, 377, 381, 384, 391, 398, 402, 
                     404, 405, 409, 412, 418, 419, 425, 426, 431, 433, 440, 447, 454, 459, 461, 465, 468, 472 , 475, 481, 482, 488, 
                     489, 494, 496, 497, 501, 503, 510, 517, 524, 531, 552, 592, 609, 731]
        self.steps_number = len(self.step_list)
        self.step = 0
        self.t = 0
        self.k = 0
        assert age_group in self._age_groups, 'age group should be one of ' + str(self._age_groups)

        self.age_group = age_group
        self.stochastic = stochastic
        self._all_internal_params_distribs = dict()
        self._all_initial_state_distribs = dict()

        # Initialize distributions of parameters and initial conditions for all regions
        self.define_params_and_initial_state_distributions()

        # Sample initial conditions and initial model parameters
        internal_params_labels = ['A', 'alpha', 'beta', 'c', 'delta', 'gamma', 'kappa', 'omega', 'p1', 'p2', 'p3', 'rho', 'sigma']

        # Define ODE model
        self.internal_model = vaccination_model

        super().__init__(internal_states_labels=['S1', 'S2', 'S3', 'S4', 'E21', 'E22', 'E23', 'E31', 'E32', 'E33', 'E41', 'E42', 'E43',
                                                 'V11', 'V21', 'V31', 'V41', 'V12', 'V22', 'V32', 'V42', 'I2', 'I3', 'I4'],
                         internal_params_labels=internal_params_labels,
                         stochastic=stochastic,
                         range_delay=range_delay)


    def define_params_and_initial_state_distributions(self):
        """
        Extract and define distributions of parameters for all age groups
        """
        for i in self._age_groups:
            self._all_internal_params_distribs[i] = dict(A=None,                                                         
                                                         alpha=[1, 2/3, 1/3, 0],
                                                         beta=[0.08, 0.04, 0.08, 0.008],
                                                         c=None,
                                                         delta=[0, 0, 0, 0.0001],
                                                         gamma=[0, 0.2, 0.1, 1/15],
                                                         kappa=[0, 1/1.5, 1/1.5, 1/1.5],
                                                         omega=[0, 1/365, 1/365, 1/365],
                                                         p1=self.p1,
                                                         p2=self.p2,
                                                         p3=self.p3,
                                                         rho=0.8944,
                                                         sigma=0#sigma_calculation(0, self.active_vaccination, self.vaccination_coverage)
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
                                                       V210=DiracDist(params=0, stochastic=self.stochastic),
                                                       V310=DiracDist(params=0, stochastic=self.stochastic),
                                                       V410=DiracDist(params=0, stochastic=self.stochastic),
                                                       V120=DiracDist(params=0, stochastic=self.stochastic),
                                                       V220=DiracDist(params=0, stochastic=self.stochastic),
                                                       V320=DiracDist(params=0, stochastic=self.stochastic),
                                                       V420=DiracDist(params=0, stochastic=self.stochastic),
                                                       I20=DiracDist(params=10/6, stochastic=self.stochastic),
                                                       I30=DiracDist(params=1/6, stochastic=self.stochastic),
                                                       I40=DiracDist(params=0, stochastic=self.stochastic)
                                                       )


    def _reset_state(self):
        """
        Resets model state to initial state.
        """
        self.current_state = dict()
        for i in self._age_groups:
            self.current_state[i] = dict(zip(self.internal_states_labels, np.array([self.initial_state[i]['{}0'.format(s)] for s in self.internal_states_labels])))


    def _get_model_params(self) -> tuple:
        """
        Get current parameters of the model

        Returns
        -------
        tuple
            tuple of the model parameters in the order of the list of labels
        """
        return tuple([self.current_internal_params[k] for k in self.internal_params_labels])


    def _get_current_state(self):
        """
        Get current state in the order of state labels.

        """
        state = []
        #print(self.current_state)
        for i in self._age_groups:
            state.append([self.current_state[i]['{}'.format(s)] for s in self.internal_states_labels])
        return state
        
    def convert_states(self, state):
        for i in state.keys():
            state[i].tolist()
        true_state = dict()
        for i in self._age_groups:
            true_state[i] = dict()
        grp=0
        for i in true_state.keys():
            for j in state.keys():
                true_state[i][j] = state[j][grp]
            grp+=1
        return true_state


    def _sample_initial_state(self):
        """
        Samples an initial model state from its distribution (Dirac distributions if self.stochastic is False).

        """
        self.initial_state = dict()
        for i in self._age_groups:
            self.initial_state[i] = dict()
            for k in self._all_initial_state_distribs[i].keys():
                self.initial_state[i][k] = self._all_initial_state_distribs[i][k].sample()

        # S10 is computed from other states, as the sum of all states equals the population size N
            self.initial_state[i]['S10'] = self.pop_size[i] - np.sum([self.initial_state[i]['{}0'.format(s)] for s in self.internal_states_labels[1:]])


    def _sample_model_params(self):
        """
        Samples parameters of the model from their distribution (Dirac distributions if self.stochastic is False).

        """
        self.initial_internal_params = dict()
        for k in self._all_internal_params_distribs['0-4'].keys():
            self.initial_internal_params[k] = self._all_internal_params_distribs['0-4'][k]
        self._reset_model_params()


    def _set_current_state(self, current_state):
        """
        Set current state to given values.

        Parameters
        ----------
        current_state: 1D nd.array
                       State the current state should be set to.

        """
        self.current_state = dict(zip(self.internal_states_labels, current_state.T))


    def run_n_steps(self, current_state=None, n=1, labelled_states=False):
        """
        Runs the model for n steps

        Parameters
        ----------
        current_state: 1D nd.array
                       Current model state.
        n: int
           Number of steps the model should be run for.

        labelled_states: bool
                         Whether the result should be a dict with state labels or a nd array.

        Returns
        -------
        dict or 2D nd.array
            Returns a dict if labelled_states is True, where keys are state labels.
            Returns an array of size (n, n_states) of the last n model states.

        """
        if current_state is None:
            current_state = self._get_current_state()
        if(self.t == self.step_list[self.step]):
            self.k, self.step = k_value(self.t, self.step)
            self.step += 1
        A_c = calculate_A_and_c(self.step, self.k, self.contact_modifiers, self.perturbations_matrices, self.transition_matrices)
        self.current_internal_params['A'], self.current_internal_params['c'] = A_c[0], A_c[1]
        # Use the odeint library to run the ODE model.
        z = odeintw(self.internal_model, current_state, np.linspace(0, 1, 2), args=(self._get_model_params()))
        self._set_current_state(current_state=z[-1].copy())  # save new current state
        self.t += 1
        self.current_state = self.convert_states(self.current_state)
        # format results
        if labelled_states:
            return self._convert_to_labelled_states(np.atleast_2d(z[1:]))
        else:
            return np.atleast_2d(z[1:])

def plotting(model, grp, cpt):
    otp = []
    for i in range(len(model)):
        otp.append(model[i][grp][cpt])
    return otp


if __name__ == '__main__':
    # Get model
    model = HeffernanOdeModel(age_group='0-4', stochastic=False)

    # Run simulation
    simulation_horizon = 1
    model_states = []
    for i in range(simulation_horizon):
        model_state = model.run_n_steps()
        model_states += model_state.tolist()

    # Plot
    time = np.arange(simulation_horizon)
    labels = model.internal_states_labels

    data = plotting(model_states, 15, 23)
    plt.plot(time, data)
    plt.show()
    # plot_stats(t=time,
    #           states=np.array(model_states).transpose(),
    #           labels=labels,
    #           show=True)

