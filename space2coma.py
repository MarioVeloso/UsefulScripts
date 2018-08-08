# space2coma.py
# Replace whitespaces from a .txt file with comas
#
# Expected command line :
# python space2coma.py input.txt output.csv
#        [0]           [1]       [2]


# Import packages
import sys

# Get command line arguments
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

# Open files
input_file = open(input_file_path,"r+")
output_file = open(output_file_path, "w")

# Replace whitespaces with comas and write output
for input_line in input_file:
    output_line = ",".join(input_line.split())
    output_line += "\n"
    output_file.write(output_line)