In this zip, there is a csv file named normalized_training which contains training data samples (after normalization). In addition, there is a csv file named normalized_testing which contains testing data samples (after normalization) of active region #12644.

The column headers for a data sample in each csv file are 
	- label, indicating whether the data sample is positive or negative;  
	- flare, representing the class of the flare within the next 24 hours where "N" indicates no flare occurs within the next 24 hours;  
	- timestamp, denoting the observation time of the data sample;  
	- NOAA, denoting the NOAA active region number;
	- HARP, denoting the corresponding HARP number;
	- 14 features where the order of the features is consistent with the order of features described in Figure 7.

This zip also contains the source code of our program, called LSTMflare.py, which is used to predict labels of testing data samples. 

The usage is given as follows:
	python3 LSTMflare.py C 0

The first argument "LSTMflare.py" is the Python program file name.
The second argument "C" denotes that the program will make predictions of >=C class flares.
The third argument "0" denotes that the program will load and use the pre-trained model, named model.h5. If one would like to re-train the model, change "0" to "1".

The output obtained by executing the above command is stored in the file named output in the zip. This output file is the same as the normalized_testing file except that it has one additional column (the first column) titled "Predicted Label", which contains labels predicted by our program using a probability threshold of 50%.

Our program is run on Python 3.6.8, Keras 2.2.4, and TensorFlow 1.12.0.





