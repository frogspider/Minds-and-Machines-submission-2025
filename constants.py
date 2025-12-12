BENEFICENCE = 0
NONMALEFICENCE = 1
JUSTICE = 2
AUTONOMY = 3

test_percentage = 0.1
one_hot_nit = False

ovr = False

bioethical_values = [BENEFICENCE, NONMALEFICENCE, JUSTICE, AUTONOMY]

n_values = len(bioethical_values)

values_names = ["BENEFICENCE", "NON-MALEFICENCE", "JUSTICE", "AUTONOMY"]
barthel_values = [10, 40, 75, 95, 100]

n_consequence_values = len([BENEFICENCE, NONMALEFICENCE])
n_duty_values = len([JUSTICE, AUTONOMY])

n_nits = 6

AGE = 0
CCD = 1
MACA = 2
EXP_SURVIVAL = 3
FRAILTY = 4
CRG = 5
NS = 6
BARTHEL = 7
LAWTON = 8
ADV_DIRECTIVES = 9
COGN_DETERIORATION = 10
EMOTIONAL = 11
DISCOMFORT = 12

AUTONOMY_UNDERSTAND = 13
AUTONOMY_INFORM = 14
AUTONOMY_COERCE = 15

n_criteria = 16
n_autonomy_criteria = 3
n_actions = 11


original_criteria_names = [
    "Age",  # 1, 2, 3, 100, 101, 102, 103, 104, 105, +99
    "CCD", # 1, 2, 3
    "MACA", # 1, 2, 3
    "Expected Survival", # 1, 2, 3
    "Frailty", # 1, 2, 3, 4
    "CRG", # 1, 2, 3, 4, 5
    "Social Support", # 1, 2, 3
    "Barthel Index",
    "Lawton Index", # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    "Advance directives", # 1, 2, 3
    "Cognitive deterioration", # 1, 2, 3, 4
    "Emotional distress", # 1, 2, 3
    "Discomfort", # 1, 2, 3, 4
    "Autonomy faculties", # 1, 2, 3
    "Autonomy Informed", # 1, 2, 3
    "Autonomy Constrained", # 1, 2, 3
]


action_names = [
    "Action CPR", # 1, ..., 11
    "Action Transplant",  # 1, ..., 11
    "Action ICU",  # 1, ..., 11
    "Action VMNI",  # 1, ..., 11
    "Action DVA",  # 1, ..., 11
    "Action Dialysis",  # 1, ..., 11
    "Action Simple",  # 1, ..., 11
    "Action Mild",  # 1, ..., 11
    "Action Advanced",  # 1, ..., 11
    "Action Palliative Surgery",  # 1, ..., 11
    "Action Curative Surgery",  # 1, ..., 11
]

max_age = 100
max_frailty = max_discomfort = 2
max_crg = max_cogn = 3
max_barthel = 100
max_lawton = 8
max_ccd = max_maca = max_ns = max_adv_directives = max_emotional = max_exp_survival = max_auto1 = max_auto2 = max_auto3 = 1

min_age = 1
min_exp_survival = min_ccd = min_maca = min_ns = min_adv_directives = min_frailty = min_crg = min_barthel = \
    min_cogn = min_discomfort = min_lawton = min_emotional = min_auto1 = min_auto2 = min_auto3 = 0
