import Patient
import Criteria
import numpy as np
from constants import action_names, n_actions, original_criteria_names, one_hot_nit, n_nits

len_nit = n_nits if one_hot_nit else 1

def translate_alignment(i, not_affected=1):

    if not_affected == 0:
        i += 1

    if i == 0:
        return "Demoted"
    elif i == 1:
        return "Not affected"
    else:
        return "Promoted"
def generational_distance(approx_solutions, true_pareto_solutions):
    """
    Compute the Generational Distance (GD) between the approximated set of solutions
    and the true Pareto-optimal set of solutions.

    Parameters:
    - approx_solutions: A 2D numpy array where each row is a solution in the approximated front.
    - true_pareto_solutions: A 2D numpy array where each row is a solution in the true Pareto front.

    Returns:
    - gd: The generational distance.
    """
    # Ensure the input arrays are numpy arrays
    approx_solutions = np.array(approx_solutions)
    true_pareto_solutions = np.array(true_pareto_solutions)

    # Initialize a list to store the minimum distances from each solution in the approximated set
    distances = []

    # For each solution in the approximated set
    for approx_solution in approx_solutions:
        # Compute the Euclidean distance to all solutions in the true Pareto set
        dists = np.linalg.norm(true_pareto_solutions - approx_solution, axis=1)
        # Find the minimum distance and store it
        distances.append(np.min(dists))

    # Convert distances list to numpy array
    distances = np.array(distances)

    # Compute the Generational Distance (GD)
    gen_dis = np.mean(distances)

    return gen_dis


# Faster than is_pareto_efficient_simple, but less readable.
def is_dominated(sol, other_sol):
    """
    Check if a solution `sol` is dominated by another solution `other_sol`.

    Parameters:
    - sol: A 1D numpy array representing the solution to be checked.
    - other_sol: A 1D numpy array representing the other solution.

    Returns:
    - True if `sol` is dominated by `other_sol`, False otherwise.
    """
    return np.all(other_sol <= sol) and np.any(other_sol < sol)

def is_pareto_efficient(costs, return_mask = True, verbose=False):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """


    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]

    if verbose:
        print(is_efficient, costs)

    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)

        identicalpoints = np.all(costs == costs[next_point_index], axis=1)

        nondominated_point_mask[identicalpoints] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points

        if verbose:
            print(is_efficient)
        costs = costs[nondominated_point_mask]

        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1


    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        print("This should never happen!")
        return is_efficient


def create_points_for_pareto(criteria, action_counter=0, model_name="real", train_model=None, verbose=False):

    point_list = list()

    if verbose:
        print("-----------------")
        print("This patient's state is : ")
        for j in range(len_nit):
            print("NIT", criteria[j])
        for j in range(len(original_criteria_names)):
            print(original_criteria_names[j], criteria[len_nit + j])
        print()
        print("The predicted alignment of each action on this patient is:")

    for action in range(n_actions):

        act_list = list()

        if model_name != "real":
            for i in range(n_actions):
                criteria[i + action_counter] = 0

            criteria[action_counter + action] = 1

            for v in range(4):  # , Values.JUSTICE, Values.AUTONOMY]

                if train_model[v] is None:
                    evaluated_action = np.random.randint(3, size=1)
                else:
                    evaluated_action = train_model[v].predict([criteria])

                act_list.append(evaluated_action[0])

            #print([a - 1 for a in act_list])
        else:
            if model_name == "real":

                patient = Patient.Patient(criteria=Criteria.Criteria(criteria, dont_normalize=True), nit=0)
                patient.set_and_apply_treatment(action, random=False)

                align_list = patient.get_alignment(apply_K=True)

                act_list = align_list

        if verbose:

            row_txt = "\multicolumn{1}{|l|}{\\textbf{" + action_names[action] + "}} "
            for j in range(len(act_list)):
                res = translate_alignment(act_list[j])
                txt = "& \multicolumn{1}{c|}{" + res + "} "

                row_txt += txt

            row_txt += "\\\ \hline"

            print(row_txt)

        point_list.append([act_list[v] for v in range(4)])


    return np.array(point_list)


def pareto_optimal_actions_per_state(criteria, action_counter=0, model_name="real", train_model=None,
                                     i_want_the_points=False, i_want_everything=False, verbose=False):

    align_points = create_points_for_pareto(criteria, action_counter, model_name, train_model, verbose=verbose)
    pareto_points = is_pareto_efficient(-1 * align_points)
    unique_front = np.unique(np.array(align_points[pareto_points]), axis=0) - 1

    if i_want_everything:

        if verbose:
            if len(unique_front) > 1:
                print()
                print("Proposed pareto front : ",  unique_front, len(unique_front), pareto_points)

                print("--- Thus, BPR recommends to do the following for this patient ---")

                for i in range(len(pareto_points)):
                    if pareto_points[i]:
                        print(action_names[i], ": permitted.")
                    else:
                        print(action_names[i], ": forbidden.")
                print("---- -----")

        return unique_front, np.array(pareto_points)

    if i_want_the_points:
        return unique_front
    return np.array(pareto_points)


if __name__ == "__main__":

    v = [[0.394, 0.295, 0.236, 1.000],
         [0.293, 0.197, 0.242, 1.000],
         [0.297, 0.125, 0.249, 1.000],
         [-1.0, -1.0, -1.0, -1.0]]

    v = np.array(v)

    pf = is_pareto_efficient(-v, return_mask=False)

    print(pf)
