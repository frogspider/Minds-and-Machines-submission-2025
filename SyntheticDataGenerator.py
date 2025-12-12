from Patient import Patient
from Actions import Action
import numpy
import random
import os
from constants import *

patients_condition = None


all_patients = list()

value_names = ["beneficence", "non-maleficence", "justice", "autonomy"]

def generate_synth_set(n_patients):


    try:
        print("This will delete the old one!")
        os.remove(file_name)
    except:
        print("First time creating synth data set!")


    v_counter = [[0, 0, 0] for _ in range(n_values)]

    for i in range(n_patients):
        patient = Patient()
        patient_list = patient.list(normalized=True)
        action = Action(random=True, maximum_actions=1)
        patient_list += action.list()
        all_patients.append(patient_list)
        patient.set_and_apply_treatment(action, random=False)
        post_list = patient.list(normalized=True)[1:]

        actual_post_list = [post_list[CCD], post_list[MACA], post_list[EXP_SURVIVAL], post_list[FRAILTY],
                            post_list[LAWTON], post_list[COGN_DETERIORATION], post_list[EMOTIONAL], post_list[DISCOMFORT]]

        align_list = patient.get_alignment(apply_K=True)

        for v in range(n_values):
            v_counter[v][align_list[v] + 1] += 1

        patient_list += actual_post_list + align_list
        patient_list = numpy.asarray(patient_list)

        with open(file_name, "ab") as f:
            #print(i)
            if i == 0:
                header_list = patient.get_state_names() + action.get_names() + patient.get_state_names(only_post=True) + value_names
                separator = ','
                patient_header = separator.join(header_list)
                #print("header : ", patient_header)
                numpy.savetxt(f, [patient_list], delimiter=",", header=patient_header, comments='')
            else:
                numpy.savetxt(f, [patient_list], delimiter=",")

    v_counter = numpy.array(v_counter) / n_patients
    min_v_counter = numpy.min(v_counter)
    min_threshold = 0.23
    print("Finally, we check that this is a balanced dataset : ", min_v_counter > min_threshold, v_counter)

    assert min_v_counter > min_threshold, "The dataset is not balanced enough: We strongly recommend generating it again."

if __name__ == "__main__":

    print("---- We generate a new synth data set -----")

    file_name = "datasets/Synth_set.csv"

    generate_synth_set(n_patients=1500)

    print("---- Generation finished. Next, we load the synth dat set -----")


    data = [numpy.loadtxt(file_name, delimiter=',', skiprows=1)][0]

    print("---- Loading finished ------")


    print("---- Now let's investigate the look of a particular patient -----")

    n_post_criteria = 8

    for i in range(4, 5):
        nit = data[i][0]
        criteria = data[i][1:n_criteria+1]
        actions = data[i][n_criteria+1:n_criteria+1+n_actions]
        postcriteria = data[i][n_criteria+1+n_actions:n_criteria+n_post_criteria+1+n_actions]
        useful_actions = data[i][n_criteria+n_post_criteria+1+n_actions:]
        print("NIT level : ", nit)
        print("Initial state : ", criteria)
        print("Treatment applied : ", actions)
        print("Final state : ", postcriteria)
        print("Alignment for each value : ", useful_actions)