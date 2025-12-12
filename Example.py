import numpy as np
from Experiments import evaluate_align_classifier

SYNTH = "datasets/Synth_set.csv"
dataset = SYNTH
data = [np.loadtxt(dataset, delimiter=',', skiprows=1)][0]
id_agent = 17

alignS, alignA, alignS_prime, also_NIT = True, True, False, True

evaluate_align_classifier(data, 1, alignS, alignA, alignS_prime, also_NIT, True, True,
                          set_name=dataset, xid="", example=id_agent)

print("Example finished.")