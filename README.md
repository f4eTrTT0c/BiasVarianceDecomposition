# Bias--Variance Decomposition for Side Channel Analysis
This repository contains code related to the 'Bias-variance Decomposition in Machine Learning-based Side-channel analysis' paper. Scripts for the following purposes are attached:
  - Train and evaluate classifiers, changing either the:
	  - Number of features
	  - Complexity of the machine learning model
	  - Number of traces used for profiling
  - Use these classifiers to predict classes and class probabilities
  - Compute Domingos bias--variance decomposition for multiclass 0--1 loss [1]
  - Compute Guessing Entropy bias--variance decomposition
  - Plot and save the results of both decompositions
  - The following classifiers are supported:
	  - Random Forest
	  - MLP
	  - CNN

**However, this repository does not include any data.** Please download any of your favorite SCA datasets (e.g. DPAv4, AES_HD, AED_RD, ASCAD, ...) and adjust the code as follows:
  - In **load_data.py**, implement the following functions:
	  - **load_data()** should return 6 numpy arrays: (*x_train, y_train, x_validate, y_validate, x_test, y_test*)
		  - *x_train* denotes the profiling traces, which should be an *N* x *p* numpy array if the number of training traces is *N*, which each tracing having *p* features;
		  - *y_train* denotes the profiling labels, based on any leakage model (e.g. HW or intermediate value), and should be a 1-dimensional array consisting of *N* integers.
		  - *x_validate, y_validate, x_test*, and *y_test* work analogously, for the validation and test set.
		  - Features are expected to be in chronologic order, so that the CNN can apply its convolution to the data.
	  - **load_data_selected()** should return 6 numpy arrays: (*x_train, y_train, x_validate, y_validate, x_test, y_test*)
		  - Here, features are expected to be sorted to some measure, the first being the most distinctive. E.g., *x_train[:, 0:10]* should contain the ten most important features for all training samples.
		  - When **load_data()** is implemented, this data can be generated by running the **select_data.py** script.
		  - Any number of features can be selected; our setting used 200 featured by default for the Random Forest and CNN.
	  - **key_byte()** should return an integer in range [0, 255] indicating the correct key byte of the test set.
	  - **key_guesses()** should return a numpy array of integers of size *M* x 256, where *M* is the size of the test set. The columns in this array indicate key candidates (i.e., column 0 corresponds to key byte 0). The values in the array should correspond with the leakage label for each trace in the test set.
	  - **n_classes()** should return an integer value, indicating the number of classes (e.g. 9 for HW model, 256 for intermediate value model). This is used by the bootstrapping algorithm to check all classes are represented when generated bootstrap samples of the training set.

## Installation
Experiments were run on Python 3.6 with scikit-learn 0.20.3, tensorflow-gpu 1.13.1 and keras 2.2.4. In principle, we encourage to always use the latest version. A Python 3.6 environment should contain the following packages:
`pip install numpy`
`pip install pandas`
`pip install scikit-learn`
`pip install tensorflow` (or: `pip install tensorflow-gpu` when running on GPU)
`pip install keras`

## References
[1] Domingos,P.: A unified bias-variance decomposition and its applications. In: Langley, P. (ed.) Proceedings of the Seventeenth International Conference on Machine Learning (ICML 2000), Stanford University, Stanford, CA, USA, June 29–July 2, 2000, pp. 231–238. Morgan Kaufmann (2000)