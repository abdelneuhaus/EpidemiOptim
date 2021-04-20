from utils import *
import numpy as np
import pandas as pd

# Vaccination or not (1 or 0)
_vaccineFull = pd.read_excel(r'./data/ScenarioPlanFranceOne.xlsx', sheet_name='vaccinateFull', skiprows=0, usecols=(1,1)).values.tolist()
vaccineFull = [x for y in _vaccineFull for x in y]
timeBreaks = [0, 71, 73, 76, 153, 173, 185, 201, 239, 244, 290, 295, 303, 305, 349, 353, 369, 370, 377, 381, 384, 391, 398, 402, 
404, 405, 409, 412, 418, 419, 425, 426, 431, 433, 440, 447, 454, 459, 461, 465, 468, 472 , 475, 481, 482, 488, 
489, 494, 496, 497, 501, 503, 510, 517, 524, 531, 552, 592, 609]#, 731]
# VOC incressed infectivity
vocInfect = 0.5

# VOC
_vocpercent = pd.read_excel(r'./data/ScenarioPlanFranceOne.xlsx', sheet_name='VOC France', skiprows=0, usecols=(3,3)).values.tolist()
vocpercent = [ x for y in _vocpercent for x in y if str(x) != 'nan']

# % incress in infectivity from VOC
def nu(t):
    return vocInfect*vocpercent[t]/100

# Coverage in the population
_coverage = pd.read_excel(r'./data/ScenarioPlanFranceOne.xlsx', sheet_name='coverage', skiprows=0, usecols=(1,1)).values.tolist()
coverage = [x for y in _coverage for x in y]

# Vaccination rate
def sigma_calculation(step):
    if (vaccineFull[step] == 1):
        return coverage[step]/100*(1/12)/(1-coverage[step]/100)
    return 1e-20
