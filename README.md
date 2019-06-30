This is an implementation of IRep++ algorithm designed to predict wine color based on wine's physicochemical variables.
============================================

This project was done as a Warsaw University of Technology assignment for Machine Learning class.

IRep++ Specification:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.369.2288&rep=rep1&type=pdf

Wine Quality Data Set:
https://archive.ics.uci.edu/ml/datasets/Wine+Quality

This implementation is based on:
https://github.com/azampagl/ai-ml-ripperk

The script (src/irep.py) handles two phases, learning and classifying, which are described in more detail below.

============================================
Arguments for python irep.py
============================================

	The following parameters are required:

-e: the execution method (learn|classify)
-a: the attribute file location.
-c: the defining attribute in the attribute file (a.k.a what we are trying to predict).
-t: the training/testing file location.
-m: the model file (machine readable results).
-o: the output file (human readable results).

============================================
Learning
============================================

	The learning part of the script reads in training cases and builds a rule model.

	It is crucial that "learn" is passed as the -e parameter (-e learn)!  The script also requires the attribute file (-a), the defining attribute (what the script is trying to predict) (-c), and the training file (-t).

	Once the rule model is built, the script will output a machine-readable data model file (-m) and a human-readable text file (-o).  The machine-readable file will be used during classification.  The human-readable file will contain an IF ... THEN ... ELSE structure with primitive equalities so that the user can easily read the rules.

============================================
Classifying
============================================

	The classifying part of the script reads in test cases and tries to accurately predict them based on the rule model learned in the learning phase.

	It is crucial that "classify" is passed as the -e parameter (-e classify)!  The script also requires the attribute file (-a), the defining attribute (what the script is trying to predict) (-c), the testing file (-t), and the machine-readable data model file created during the learn phase (-m).

	Once the test cases are analyzed, the results will be put into a human-readable text file (-o).

======================
	Usage
======================

	The following are some example use cases.

Classifying test cases for the restaurant scenario.
> python irep.py -e learn -a "./attributes/wine.txt" -c color -t "./train/train.txt" -m "./results/wine-model.dat" -o "./results/wine-model.txt"

Classifying test cases for the id scenario.
> python irep.py -e classify -a "./attributes/wine.txt" -c color -t "./test/test.txt" -m "./results/wine-model.dat" -o "./results/wine-model-test.txt"
