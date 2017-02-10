Authors: Jason Deutsch and 	Sreeraj Rimmalapud
Project 1: MSapriori algorithm implementation

- This program performs the MSapriori algorithm and is written in Python.

Instructions:
For ease of use, keep all files in one directory. By simply running the file
"gen_output.py" the program will run as is and will return the frequent itemsets
as a text file entitled "output-patterns.txt." The output file name can be changed 
on line 6 of "gen_output.py" if desired.

Algorithm implementation is performed in "ms_apriori.py" and requires two input files - 
an input file and a parameter file. The input file must contain the transaction database 
and the parameter file must contain the MIS values, cannot_be_together sets, must-have sets,
and the support difference constraint (SDC). By default, the input file should be labeled
as "input-data.txt" and the parameter file should be labeled as "parameter-file.txt". 
If these files are not named as mentioned the input file names can be read by the program
by changing line 158 and line 159 in the file "ms_apriori.py", for the input data and parameter file
respectively. 


