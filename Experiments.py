from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from constants import (n_values, n_criteria, n_actions, n_nits, test_percentage, BENEFICENCE, NONMALEFICENCE,
                        one_hot_nit, values_names, original_criteria_names, action_names)
import pickle
import math
import os

from copy import deepcopy
from pareto_front import generational_distance, pareto_optimal_actions_per_state

logistic_n = True
len_nit = n_nits if one_hot_nit else 1


classifier_names =  [
    "Logistic Regression",
    "KNN",
    "SVC with linear kernel",
    "SVC with rbf kernel",
    "Gaussian Process",
    "Decision Tree",
    "Random forest",
    "MLP",
    "QDA",
    "Gradient Boosting",
    "XG Boosting",
    "Random dummy"
]


LOG_REGRESSION = 0
XG_BOOST = 10

classifiers = [
    LogisticRegression(random_state=42, max_iter=10000),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=4, random_state=42),
    RandomForestClassifier( max_depth=5, n_estimators=10, max_features=1, random_state=42),
    MLPClassifier(alpha=1, max_iter=3000, random_state=42),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier(),
    XGBClassifier(),
]


train_data_eval = False

#print("Evaluate with train data : ", train_data_eval)


n_future_criteria = 8

y_max = 1.0
y_min = -1.0

delta_y = y_max - y_min


if test_percentage == 0:
    classifiers = [classifiers[0]]
    classifier_names = [classifier_names[0]]

NIT_names = ["NIT"]

if one_hot_nit:
    NIT_names = [
        "NIT 1",
        "NIT 2",
        "NIT 2B",
        "NIT 3",
        "NIT 4",
        "NIT 5",
    ]


post_criteria_names = [
    "Post CCD",
    "Post MACA",
    "Post Exp Survival",
    "Post Frailty",
    "Post Independence",
    "Post Cognitive",
    "Post Emotional",
    "Post Discomfort",
]


def logistic(b0, b1, x):
    return 1. / (1 + math.exp(-(b0+b1*x)))


def configure_criteria_names(aliS, aliA, aliS_prime, include_NIT):
    criteria_names = []
    for n in original_criteria_names:
        criteria_names.append(n)

    included_criteria_names = criteria_names if aliS else []
    included_action_names = action_names if aliA else []
    included_post_criteria_names = post_criteria_names if aliS_prime else []

    criteria_names = included_criteria_names + included_action_names + included_post_criteria_names

    if include_NIT:
        criteria_names = NIT_names + criteria_names

    return criteria_names

def predict_pareto_front(raw_dataset, train_model, verbose=False, example=-1):

    y_pred = list()
    y_test = list()

    gen_dis = list()

    tn, fp, fn, tp = 0, 0, 0, 0


    for i in range(len(raw_dataset)):

        example_time = train_model[0] is not None and i == example

        patient_state = raw_dataset[i]
        precriteria = patient_state[len_nit:n_criteria + len_nit]
        proposed_pareto_front, proposed_pareto_front_idx = pareto_optimal_actions_per_state(patient_state, action_counter=len_nit + n_criteria,
                                                                 model_name="Predict", train_model=train_model,
                                                                 i_want_everything=True, verbose=example_time)

        if example > -1 and len(proposed_pareto_front) > 1 and train_model[0] is not None:
            print("Change id_agent to nª ", i, " to observe another interesting example.")

            #print("Proposed pareto front : ", proposed_pareto_front, proposed_pareto_front_idx)
        correct_pareto_front, correct_pareto_front_idx = pareto_optimal_actions_per_state(precriteria, i_want_everything=True, verbose=verbose)

        #print("Moment of truth : ", optimal_alignment, proposed_optimal_alignment)
        y_pred.append(proposed_pareto_front)
        y_test.append(correct_pareto_front)


        gen_dis.append(generational_distance(proposed_pareto_front, correct_pareto_front))

        confusion_matrix_i = confusion_matrix(correct_pareto_front_idx, proposed_pareto_front_idx, labels=[False, True])

        tn_i, fp_i, fn_i, tp_i = confusion_matrix_i.ravel()

        tn += tn_i
        fp += fp_i
        fn += fn_i
        tp += tp_i


        if verbose:
            print("Real pareto front : ", correct_pareto_front)
            print("Generational distance : ", gen_dis[i])
            print("----")

    if verbose:
        print("Average Generational distance: ", np.mean(gen_dis))

        print("Confusion matrix : ", tn, fp, fn, tp)

    confusion_matrix_data = np.array([tn, fp, fn, tp])
    return np.mean(gen_dis), confusion_matrix_data


def train_classifier_for_pareto(i, features, all_labels, random_split):

    models = list()

    for v in range(n_values):

        labels = all_labels[:, v]

        labels = LabelEncoder().fit_transform(labels)

        if test_percentage > 0:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_percentage,
                                                                    random_state=random_split)


        else:
            X_train = features
            y_train = labels

        if i < len(classifiers):
            if v in [BENEFICENCE, NONMALEFICENCE]:
                model = classifiers[XG_BOOST].fit(X_train, y_train)
            else:
                model = classifiers[LOG_REGRESSION].fit(X_train, y_train)

        else:
            model = None

        models.append(deepcopy(model))


    return models, X_test


def evaluate_classifier(i, features,labels, evaluate_accuracy, silent=True, random_split=-1, only_model=False):

    weights = []
    indep_term = None

    if not silent:
        print("Model n ", i)
        print("----Training-----")

    if test_percentage > 0:
        if random_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_percentage, random_state=random_split)
        else:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_percentage)
    else:
        X_train = features
        y_train = labels

    if i < len(classifiers):
        model = classifiers[i].fit(X_train, y_train)
    else:
        model = None

    if only_model:
        return model

    if test_percentage > 0:
        if i < len(classifiers):

            if train_data_eval:
                y_pred = model.predict(X_train)
            else:
                y_pred = model.predict(X_test)

        else:
            if train_data_eval:
                y_pred = np.random.randint(3, size=len(X_train))
            else:
                y_pred = np.random.randint(3, size=len(X_test))


        if not silent:
            print("---Testing----")

        if train_data_eval:
            if evaluate_accuracy:
                acc = accuracy_score(y_pred, y_train)
            else:
                acc = f1_score(y_pred, y_train, average="macro")
        else:
            if evaluate_accuracy:
                acc = accuracy_score(y_pred, y_test)
            else:
                acc = f1_score(y_pred, y_test, average="macro")

        if not silent:

            mat = confusion_matrix(y_pred, y_test)
            print("Confusion matrix : ", mat)

            print()

            print("Accuracy : ", acc)
    else:

        if i < len(classifiers):
            print("Yeah")
#            weights = model.coef_
#            indep_term = model.intercept_

#            print("Intercept : ", indep_term)

        acc = 0

    return acc, model, weights, indep_term


def evaluate_align_classifier(train_data, max_iterations, aliS, aliA, aliS_prime, include_nit, evaluate_accuracy, 
                              evaluate_pareto, set_name, xid, example=False):

    criteria_names = configure_criteria_names(aliS, aliA, aliS_prime, include_nit)

    if not example:
        try:
            os.remove("results/output"+str(xid)+".txt")
            print("This experiment ID is not new. The old output file for this experiment has been deleted.")
        except:
            print("This experiment ID is new.")

        with open("results/output"+str(xid)+".txt", "a") as f:
            print("--- For this experiment, we trained the algorithms with the following configuration --- ", file=f)
            print("Dataset loaded : ", set_name, file=f)
            print("NIT criterion data used for training: ", include_nit, file=f)
            print("Criteria data used for training : ", aliS, file=f)
            print("Action data used for training: ", aliA, file=f),
            print("Post-criteria data used for training: ", aliS_prime, file=f)

            if evaluate_pareto:
                metric = "generational distance + confusion matrix"
            elif evaluate_accuracy:
                metric = "accuracy"
            else:
                metric = "f1-score"

            print("Computed metric: ", metric, file=f)

            print("--- ---", file=f)

    all_weights = [[] for _ in range(n_values)]
    all_indies = [0 for _ in range(n_values)]

    tested_algoritms = [["" for _ in range(n_values)] for _ in range(len(classifiers) + 1)]

    tested_algoritms_pareto = [-1 for _ in range(len(classifiers) + 1)]
    tested_algoritms_confusion = [-1 for _ in range(len(classifiers) + 1)]

    X = []
    Y_v = []
    provisional_results = np.zeros((len(classifiers) + 1, n_values, max_iterations))
    provisional_results_pareto = np.zeros((len(classifiers) + 1, max_iterations))

    provisional_results_confusion = np.zeros((len(classifiers) + 1, max_iterations, 4))

    #print("length data", len(train_data))
    for i in range(len(train_data)):
        nit = train_data[i][0:len_nit]

        criteria = train_data[i][len_nit:n_criteria + len_nit]
        actions = train_data[i][n_criteria + len_nit:n_criteria + len_nit + n_actions]
        postcriteria = train_data[i][n_criteria + len_nit + n_actions:n_criteria + len_nit + n_actions + n_future_criteria]
        alignments = train_data[i][n_criteria + len_nit + n_actions + n_future_criteria:]

        """
        if i == 5:
            for j in range(len_nit):
                print(NIT_names[j], nit[j])
            print()
            for j in range(len(criteria)):
                print(criteria_names[len_nit + j], criteria[j])
            print()
            for j in range(len(actions)):
                print(action_names[j], actions[j])
            print()
            for j in range(len(post_criteria_names)):
                print(post_criteria_names[j],postcriteria[j])
            print()
            print(alignments)
        """

        if aliA and not aliS and not aliS_prime:
            x_vars = actions
        elif not aliA and aliS and not aliS_prime:
            x_vars = criteria
        elif not aliA and not aliS and aliS_prime:
            x_vars = postcriteria
        elif aliA and aliS and not aliS_prime:
            x_vars = np.concatenate((criteria, actions), axis=0)
        elif not aliA and aliS and aliS_prime:
            x_vars = np.concatenate((criteria, postcriteria), axis=0)
        else:
            x_vars = np.concatenate((criteria, actions), axis=0)
            x_vars = np.concatenate((x_vars, postcriteria), axis=0)

        if include_nit:
            x_vars = np.concatenate((nit, x_vars), axis=0)
        X.append(x_vars)

        Y_v.append(alignments)

    X = np.array(X)
    Y_v = np.array(Y_v)


    le = LabelEncoder()

    len_reg = len(classifiers) + 1

    if test_percentage == 0:
        len_reg -= 1

    if example:
        print("Model finished trained. Now we can start doing predictions.")

    TN = 0
    FP = 1
    FN = 2
    TP = 3
    if evaluate_pareto:
        for k in [0, len_reg - 1] : #range(len_reg):
            for it in range(max_iterations):

                print("Proceeding with experiment n ", it, " / ", max_iterations)


                pareto_model, test_set = train_classifier_for_pareto(k, X, Y_v, random_split=it)

                #print("length test set", len(test_set), len(X))
                provisional_results_pareto[k, it], provisional_results_confusion[k, it] = predict_pareto_front(test_set, pareto_model, example=example)


                total_positives = provisional_results_confusion[k, it, TP] + provisional_results_confusion[k, it, FN]
                total_negatives = provisional_results_confusion[k, it, TN] + provisional_results_confusion[k, it, FP]


                if total_positives > 0:
                    provisional_results_confusion[k, it, TP] /= total_positives
                    provisional_results_confusion[k, it, FN] /= total_positives

                if total_negatives > 0:
                    provisional_results_confusion[k, it, TN] /= total_negatives
                    provisional_results_confusion[k, it, FP] /= total_negatives

            #print(k, provisional_results_confusion[k,:,1])
            results_std = np.round(provisional_results_pareto[k].std(), 3)
            results_mean = np.round(provisional_results_pareto[k].mean(), 3)
            tested_algoritms_pareto[k] = str(results_mean) + " + " + str(results_std)

            confusion_std = [np.round(provisional_results_confusion[k,:,i].std(), 3) for i in range(4)]
            confusion_mean = [np.round(provisional_results_confusion[k,:,i].mean(), 3) for i in range(4)]
            tested_algoritms_confusion[k] = str(confusion_mean) + " + " + str(confusion_std)
        for k in [0, len_reg - 1]:
            if not example:
                with open("results/output" + str(xid) + ".txt", "a") as f:
                    print("Generational distance for classifier ", classifier_names[k], ": ", tested_algoritms_pareto[k], file=f)
                    print("Confusion matrix  for classifier ", classifier_names[k], ": ",  tested_algoritms_confusion[k], file=f)
                    print("The order in confusion matrix is : true negatives, false positives, false negatives, true positives (all normalised) ", file=f)

    else:


        for v in range(n_values):

            print("============== COMPUTING FOR VALUE ", values_names[v], "  =================")

            y = Y_v[:, v]

            y = le.fit_transform(y)


            for k in range(len_reg):
                for it in range(max_iterations):
                    print("Proceeding with experiment n ", it + 1, " / ", max_iterations, " for classifier ", k + 1, " / ", len_reg)

                    r, model, w, i = evaluate_classifier(k, X, y, evaluate_accuracy)

                    if False: #test_percentage == 0:
                        with open('model' + str(v) + '.pkl', 'wb') as file:
                            pickle.dump(model, file)

                    provisional_results[k, v, it] = r

                results_std = np.round(provisional_results[k, v].std(), 3)
                results_mean = np.round(provisional_results[k, v].mean(), 3)
                tested_algoritms[k][v] = str(results_mean) + " + " + str(results_std)

        with open("results_alignSAS.pkl", "wb") as fp:  # Pickling
            pickle.dump(tested_algoritms, fp)

    if not evaluate_pareto:
        with open("results_alignSAS.pkl", "rb") as fp:  # Unpickling
            loaded_object = pickle.load(fp)

        if test_percentage > 0:
            for i in range(len(loaded_object)):
                row_txt = "\multicolumn{1}{|l|}{\\textbf{"+ classifier_names[i] + "}} "
                for j in range(len(loaded_object[i])):

                    res = loaded_object[i][j]

                    res = res.split("+")

                    txt = "& $" + res[0] + "\pm" + res[1] + "$ "

                    row_txt += txt

                row_txt += "\\\ \hline"

                if not example:
                    with open("results/output" + str(xid) + ".txt", "a") as f:
                        print(row_txt, file=f)
        else:
            all_weights = np.array(all_weights)

            n_features = len(all_weights[0])

            #print("---")
            #print(len(criteria_names), n_features)
            #print(criteria_names)
            #print("---")
            for i in range(n_features):

                row_txt = "\multicolumn{1}{|l|}{\\textbf{" + criteria_names[i] + "}} "

                row_weights = all_weights[:,i]

                for w in row_weights:

                    if logistic_n:

                        final_w = logistic(0, w, 1)
                    else:
                        final_w = w
                    row_txt += "& " + str(np.round(final_w,3))

                row_txt += "\\\ \hline"

                if not example:
                    with open("results/output" + str(xid) + ".txt", "a") as f:
                        print(row_txt, file=f)

            row_txt = "\multicolumn{1}{|l|}{\\textbf{" + "Independent term" + "}} "

            for i in all_indies:
                row_txt += "& " + str(np.round(i, 3))

            row_txt += "\\\ \hline"

            if not example:
                with open("results/output" + str(xid) + ".txt", "a") as f:
                    print(row_txt, file=f)
    try:
        os.remove("results_alignSAS.pkl")
    except:
        pass

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    HOSPITAL = "datasets/Hospital_set.csv"
    SYNTH = "datasets/Synth_set.csv"
    alignS, alignA, alignS_prime, also_NIT = True, True, False, True
    max_it = 20

    dataset_names = ["SYNTH_fullset", "HOSPITAL", "SYNTH_fullset"]
    metric_names =["_f1score", "_pareto", "_accuracy"]

    for d in dataset_names:
        for m in metric_names:

            print("Now the experiment with " + d + " and measuring : " + m)

            if d != "SYNTH_fullset" and m == "_pareto":
                continue

            experiment_id = d + m

            if d == "HOSPITAL":
                dataset = HOSPITAL
                data = [np.loadtxt(dataset, delimiter=',', skiprows=1)][0]
            elif d == "SYNTH_subset":
                dataset = SYNTH
                data = [np.loadtxt(dataset, delimiter=',', skiprows=1)][0][:150]
            else:
                dataset = SYNTH
                data = [np.loadtxt(dataset, delimiter=',', skiprows=1)][0]

            if m == "_accuracy":
                metric_accuracy = True
                metric_pareto = False
            elif m == "_f1score":
                metric_accuracy = False
                metric_pareto = False
            else:
                metric_accuracy = metric_pareto = True

            evaluate_align_classifier(data, max_it, alignS, alignA, alignS_prime, also_NIT, metric_accuracy, metric_pareto,
                                      set_name=dataset, xid=experiment_id)

            print("Experiment finished. Please read results at file : output" + str(experiment_id) + ".txt")
