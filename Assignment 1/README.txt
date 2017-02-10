Authors: Jason Deutsch and 	Sreeraj Rimmalapud
Project 1: MSapriori algorithm implementation

- This program performs the MSapriori algorithm and is written in Python.

Instructions:
For ease of use, keep all files in one directory. 

Running the file:
Run the gen_output.py file and provide it with the following command line arguments

python gen_output.py <input filename> <parameter filename> <output filename>

*IMP - prints out an error if any of the arguments are not provided


Algorithm implementation is performed in "ms_apriori.py" and requires two input files - 
an input file and a parameter file. The input file must contain the transaction database 
and the parameter file must contain the MIS values, cannot_be_together sets, must-have sets,
and the support difference constraint (SDC).  
