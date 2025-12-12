Needed libraries:

- Numpy
- Scikit-Learn

Folders:
- Datasets folders includes the two datasets used in our experiments: the Hospital set and the Synth set.
- Results folders provides the results of all five experiments explained in the main paper.

Files:

- SyntheticDataGenerator.py : execute it to generate a completely new Synth set.

- Experiments.py : execute it to repeat all experiments in Section 5 of the paper.

- Example.py : execute it to repeat the example application of our BPR algorithm to a patient in Section 6 of the paper.

The rest of python files are auxiliary for these three:

- constants.py contains variables necessary for all other files.
- Alignments.py / Actions.py / Criteria.py / Patient.py contain the logic necessary to create synthetic patients.
- pareto_front.py contains the logic necessary for computing metrics associated with the Pareto front in our experiments.