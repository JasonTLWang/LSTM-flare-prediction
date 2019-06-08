This zip contains one folder, named C, containing data samples in the >=C class. In the folder, there are three csv files named normalized_training, normalized_validation, normalized_testing respectively, which contain training, validation and testing data samples (after normalization).
The difference between these three csv files and those of Table 2 is that these three csv files contain incomplete data samples that have at least one missing physical feature value. After removing these incomplete data samples, one would get the same data samples as described in Table 2. 

This zip also contains the source code of our program, called LSTMpredict.py, which is used to calculate the performance metric values for the >=C class as described in Table 5. Our program will (i) automatically remove the incomplete data samples in the three csv files; (ii) apply the zero-padding strategy; (iii) select and use the 14 most important features for the >=C class; (iv) perform the cross-validation test as described in the paper.

The usage is given as follows:
	python3 LSTMpredict.py C

The first argument "LSTMpredict.py" is the Python program file name.
The second argument "C" denotes that the program will run for the >=C class.

The output obtained by executing the above command is stored in the file named output in the zip. This output file contains the performance metric values calculated by our program using the optimal threshold that maximizes TSS.

Our program is run on Python 3.6.8, Keras 2.2.4, and TensorFlow 1.12.0.





