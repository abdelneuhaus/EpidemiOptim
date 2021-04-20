from epidemioptim.utils import *
import numpy as np
import pandas as pd


PATH_TO_DATA = get_repo_path() + '/data/jane_model_data/ScenarioPlanFranceOne.xlsx'
PATH_TO_HOME_MATRIX = get_repo_path() + '/data/jane_model_data/contactHome.txt'
PATH_TO_SCHOOL_MATRIX = get_repo_path() + '/data/jane_model_data/contactSchool.txt'
PATH_TO_WORK_MATRIX = get_repo_path() + '/data/jane_model_data/contactWork.txt'
PATH_TO_OTHER_MATRIX = get_repo_path() + '/data/jane_model_data/contactOtherPlaces.txt'
PATH_TO_COMORBIDITY_MATRIX = get_repo_path() + '/data/jane_model_data/coMorbidity.txt'


# Inital population for each age class
# Need to install openpyxl
_population = pd.read_excel(PATH_TO_DATA, sheet_name='population', skiprows=3, usecols=(2,2))
_population = _population.values.tolist()
Pop = sum(_population, [])  # Population of each age class
totalPop = sum(Pop) # French population


# kval : factor that reduces the contact matrix (lockdown makes k goes from 1 to 0.8)
kval = pd.read_excel(PATH_TO_DATA, sheet_name='kvalFull')
kval = kval.iloc[0:59,1:4].values.tolist()


# Contact matrices
perMat = pd.read_excel(PATH_TO_DATA, sheet_name='Perturbation Matricies')
sf1 = perMat.iloc[2:18,1:17].values.tolist()
sf2 = perMat.iloc[21:37,1:17].values.tolist()
sf3 = perMat.iloc[40:56,1:17].values.tolist()
sf4 = perMat.iloc[59:75,1:17].values.tolist()
sf5 = perMat.iloc[78:94,1:17].values.tolist()
sf6 = perMat.iloc[97:113,1:17].values.tolist()

of1 = perMat.iloc[2:18,19:35].values.tolist()
of2 = perMat.iloc[21:37,19:35].values.tolist()

wf1 = perMat.iloc[2:18,37:53].values.tolist()
wf2 = perMat.iloc[21:37,37:53].values.tolist()


# Contact modifier
dataModifier = pd.read_excel(PATH_TO_DATA, sheet_name='contactModifiersFull')
E1 = np.matrix(dataModifier.iloc[2:63,2:5]).tolist()
E2 = np.matrix(dataModifier.iloc[2:63,6:9]).tolist()
E3 = np.matrix(dataModifier.iloc[2:63,10:13]).tolist()
E4 = np.matrix(dataModifier.iloc[2:63,14:17]).tolist()
E5 = np.matrix(dataModifier.iloc[2:63,18:21]).tolist()
Ebase = np.matrix(dataModifier.iloc[2:63,22:25]).tolist()
contactModifiers = [E1, E2, E3, E4, E5, Ebase]
