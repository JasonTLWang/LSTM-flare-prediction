There are three zip files/folders named C, M, M5 containing data samples in >=C, >=M, >=M5.0 class respectively as described in Table 2.

In each folder, there are 3 csv files named training, validation, testing respectively, which contain data samples (before normalization) as described in Table 2.
In addition, there are 3 csv files named normalized_training, normalized_validation, normalized_testing respectively, which contain data samples (after normalization) as described in Table 2.

The column headers for a data sample in each csv file are 
	- label, indicating whether the data sample is positive or negative;  
	- flare, representing the class of the flare within the next 24 hours where "N" indicates no flare occurs within the next 24 hours;  
	- timestamp, denoting the observation time of the data sample;  
	- NOAA, denoting the NOAA active region number;
	- HARP, denoting the corresponding HARP number;
	- 40 features where the order of the features is consistent with the order of features described in Figures 5, 6 and 7 for the three different flare classes.
