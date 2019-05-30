import sys 
import numpy as np 

# data loading

input = open('NER_Dataset/test_bio_one_line.txt', 'r').readlines()  
lower = open('NER_Dataset/test_bio_new.txt', "r", encoding="utf8").readlines() 

output = open('NER_Dataset/test_bio_new_lowercase.txt', "w", encoding="utf8") 


j = 0
split_lower = input[j].split(" ")

k = 0
for i in range(len(lower)):
    line = lower[i]
    if "<S>" in line:  
        output.write("<S>			<S>\n")
        continue
    else:
        if split_lower[k] == "\n":
            j += 1
            split_lower = input[j].split(" ")
            k = 0
            
            new_line = split_lower[k] + "\t" + line.split("\t")[1] + "\t" + line.split("\t")[2] + "\t" + line.split("\t")[3] 
            output.write(new_line)
            
        else:
            new_line = split_lower[k] + "\t" + line.split("\t")[1] + "\t" + line.split("\t")[2] + "\t" + line.split("\t")[3] 
            output.write(new_line)
        k += 1    
output.close()