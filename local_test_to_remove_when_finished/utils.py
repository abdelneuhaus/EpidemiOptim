def get_text_file_data(path):
    """
    Read a txt file and convert each line into a sublist
    Output : a list with as many sublists as lines in the file
    """
    file = open(path, "r")
    tmp = []
    for line in file:
        stripped_line = line.strip()
        line_list = stripped_line.split()
        line_list = [float(x) for x in line_list]
        tmp.append(line_list)
    file.close()
    return tmp


def create_list(value, sublist_nb, sublist_size):
    """
    Create a list of len sublist_size, filled with sublist_nb sublists. Each sublist is filled with the value value
    """
    out = []
    tmp = []
    for i in range(sublist_nb):
        for j in range(sublist_size):
            tmp.append(value)
        out.append(tmp)
        tmp = []
    return out


def duplicate_data(data, nbr):
    """
    Duplicate a single list (data) into identical sublists a given number of times (nbr)
    """
    out = []
    for i in range (nbr):
        out.append(data)
    return out


def generate_age_output(res, compartment, age, numberC):
    """
    Get the data for a run of odeint:
    compartment: S, E1, E2, E3, V1, V2 or I
    age: one of the 16 groups
    numberC: which subcompartiment (for example, S1, S2, S3 or S4)
    """
    ageX = []
    for i in range(len(res)):
        ageX.append(res[i][compartment][age][numberC])
    return ageX
    
    