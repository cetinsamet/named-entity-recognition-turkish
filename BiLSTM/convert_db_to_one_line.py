import sys 
import numpy as np 
import random

# data loading


raw_txt_file = open('NER_Dataset/test_bio_new.txt', 'r', encoding="utf8" ).readlines()  
file = open('NER_Dataset/test_bio_one_line.txt', "w") 




for line in raw_txt_file:
    if "<S>" in line:  
        pass
    else:
        file.write(line.split("\t")[0] + " ")










# counter = 0
# a = random.randint(25,35)
# for line in raw_txt_file:  
    # if line.strip() != "" :
        # file.write(line.strip() + " ")
    
    # if counter == a:
        # a = random.randint(25, 35)
        # counter = 0
        # file.write( "\n")
        
    # counter += 1
file.close()