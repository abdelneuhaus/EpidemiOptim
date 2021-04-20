from utils import *
import numpy as np
import pandas as pd
from vaccinationParameters import timeBreaks


# Age
age = np.arange(2.5, 80, 5).tolist()
N = 1
age_interval = np.ones(15).tolist()
age_interval = [x*5 for x in age_interval]
age_interval.append(25.0)


# Inital population for each age class
_population = pd.read_excel(r'./data/ScenarioPlanFranceOne.xlsx', sheet_name='population', skiprows=3, usecols=(2,2))
population = _population.values.tolist()[0]


# kval
kval = pd.read_excel(r'./data/ScenarioPlanFranceOne.xlsx', sheet_name='kvalFull')
kval = kval.iloc[0:59,1:4].values.tolist()


# Vaccine efficacy
rho = pd.read_excel(r'./data/ScenarioPlanFranceOne.xlsx', sheet_name='rho', skiprows=2)
rho = rho.iloc[[0]].values.tolist()
rho = rho[0]


# Contact matrices
W = get_text_file_data('./data/contactWork.txt')[0] # work
O = get_text_file_data('./data/contactOtherPlaces.txt')[0] # other places
H = get_text_file_data('./data/contactHome.txt')[0] # home
S = get_text_file_data('./data/contactSchool.txt')[0] # school
p1 = get_text_file_data('./data/coMorbidity.txt')[0] # comorbidity

sf = [1]
wf = [1]
of = [1]

perMat = pd.read_excel(r'./data/ScenarioPlanFranceOne.xlsx', sheet_name='Perturbation Matricies')
sf1 = perMat.iloc[2:18,1:17].values.tolist()[0]
sf2 = perMat.iloc[21:37,1:17].values.tolist()[0]
sf3 = perMat.iloc[40:56,1:17].values.tolist()[0]
sf4 = perMat.iloc[59:75,1:17].values.tolist()[0]
sf5 = perMat.iloc[78:94,1:17].values.tolist()[0]
sf6 = perMat.iloc[97:113,1:17].values.tolist()[0]

of1 = perMat.iloc[2:18,19:35].values.tolist()[0]
of2 = perMat.iloc[21:37,19:35].values.tolist()[0]

wf1 = perMat.iloc[2:18,37:53].values.tolist()[0]
wf2 = perMat.iloc[21:37,37:53].values.tolist()[0]


H1 = np.multiply(np.matrix(H).T,population)
S1 = np.multiply(np.matrix(S).T,population)
W1 = np.multiply(np.matrix(W).T,population)
O1 = np.multiply(np.matrix(O).T,population)

Hmat1 = np.multiply(np.matrix(H1), np.matrix(H1).T)
Smat1 = np.multiply(np.matrix(S1), np.matrix(S1).T)
Wmat1 = np.multiply(np.matrix(W1), np.matrix(W1).T)
Omat1 = np.multiply(np.matrix(O1), np.matrix(O1).T)

Hmat = np.divide(np.sqrt(Hmat1), population)
Smat = np.divide(np.sqrt(Smat1), population)
Wmat = np.divide(np.sqrt(Wmat1), population)
Omat = np.divide(np.sqrt(Omat1), population)


# Contact modifier
dataModifier = pd.read_excel(r'./data/ScenarioPlanFranceOne.xlsx', sheet_name='contactModifiersFull')
E1 = np.matrix(dataModifier.iloc[2:63,2:5]).tolist()
E2 = np.matrix(dataModifier.iloc[2:63,6:9]).tolist()
E3 = np.matrix(dataModifier.iloc[2:63,10:13]).tolist()
E4 = np.matrix(dataModifier.iloc[2:63,14:17]).tolist()
E5 = np.matrix(dataModifier.iloc[2:63,18:21]).tolist()
Ebase = np.matrix(dataModifier.iloc[2:63,22:25]).tolist()
contactModifiers = [E1, E2, E3, E4, E5, Ebase]


# Modify contact regarding the date
def contactModifiersComputation(t, i, k):
    sf = [1]
    wf = [1]
    of = [1]
    scenario = 0
    phase = contactModifiers[scenario]
    if (phase[i][0] == 1):
        sf = sf1
    elif (phase[i][0] == 2):
        sf = sf2
    elif (phase[i][0] == 3):
        sf = sf3
    elif (phase[i][0] == 4):
        sf = sf4
    elif (phase[i][0] == 5):
        sf = sf5
    elif (phase[i][0] == 6):
        sf = sf6

    if (phase[i][1] == 1):
        of = of1
    elif (phase[i][1] == 2):
        of = of2

    if (phase[i][2] == 1):
        wf = wf1
    elif (phase[i][2] == 2):
        wf = wf2

    USc = Hmat + np.multiply(np.matrix(wf), Wmat) + np.multiply(np.matrix(of), Omat) + np.multiply(np.matrix(sf), Smat)
    _con = np.multiply(USc, k)
    _A = _con.sum(axis=0)
    _c = np.divide(_con, _A)
    con, A, c = _con.tolist(), _A.T.tolist(), _c.tolist()[0]
    A = [i for j in A for i in j]
    return con, A, c


_p1 = p1 
p2 = _p1
p3 = _p1
p3 = np.matrix(p3).T.tolist()
p3[0] = duplicate_data(0, N)
p3 = np.matrix(p3).T.tolist()[0]

# First dose efficacy
epsilon = pd.read_excel(r'./data/ScenarioPlanFranceOne.xlsx', sheet_name='epsilon', skiprows=2)
epsilon = epsilon.iloc[[0]].values.tolist()
epsilon = epsilon[0]
epsilon = [x for x in epsilon if str(x) != 'nan']
ep = 1 - epsilon[0]


# Disease-induced mortality
# delta
delta4 = 0.00001 # only in I4
delta3 = 0
delta = [0, 0, delta3, delta4]


# Rate of progress through the pre-infectious period of infeciton
# transition from E to I
_kappa = 1.5
kappa1 = 0
kappa2 = 1/_kappa
kappa3 = 1/_kappa
kappa4 = 1/_kappa
kappa = [kappa1, kappa2, kappa3, kappa4]


# Recovery rate (per day)
gamma2 = 0.2
gamma3 = 0.1
gamma4 = 1/15
Gamma = [0, gamma2, gamma3, gamma4]


# infectivity per day
beta3 = 0.08    # or 0.02 (to search)
beta1 = 0
beta2 = 0.5*beta3
beta4 = 0.1*beta3
beta = [beta1, beta2, beta3, beta4]


# waning rate = loss of immunity with time
omega2 = 1 
omega3 = 1
omega4 = 1
omega = [0, omega2/365, omega3/365, omega4/365]


# Susceptibility (no unite)
alpha = [1, 2/3, 1/3]


# Others parameters
qq = 0.7 # first dose protection from severe disease


def get_k_value(val, j):
    _kval = np.array(kval).T.tolist()[val]
    k = None
    k = _kval[j]
    return k