from math import *
import pandas as pd
import numpy as np

# Never used
def tau_calculation(mu, q, age_interval,N):
    """
    Calculation of tau value for each age class
    mu: 
    q: a variable
    age_interval: 5
    N: number of age classes
    """
    tau = [None]*16
    for i in range(0,N):
        tau[i] = (mu[i]+q)/(exp((mu[i]+q)*365*age_interval[i])-1)
    return tau


def get_text_file_data(path):
    file = open(path, "r")
    tmp = []
    for line in file:
        stripped_line = line.strip()
        line_list = stripped_line.split()
        line_list = [float(x) for x in line_list]
        tmp.append(line_list)
    file.close()
    return tmp


def get_inital_state(start, end):
    a = range(start,end+1)
    tmp = []
    otp = []
    for i in a :
        tmp.append(i)
    
    res = []
    for i in tmp:
        otp.append(i)
        if (i%4 == 0):
            res.append(otp)
            otp = []
    return res


def create_list(value, sublist_nb, sublist_size):
    out = []
    tmp = []
    for i in range(sublist_nb):
        for j in range(sublist_size):
            tmp.append(value)
        out.append(tmp)
        tmp = []
    return out


def duplicate_data(data, nbr):
    out = []
    for i in range (nbr):
        out.append(data)
    return out


def generate_age_output(res, compartment, age):
    ageX = []
    for i in range(len(res)):
        if res[i][compartment][age] < 0:
            ageX.append(0)
        else:
            ageX.append(res[i][compartment][age])
    return ageX