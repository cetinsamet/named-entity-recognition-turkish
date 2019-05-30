import sys 
import numpy as np 

# data loading

black_list = ['"', '.', ',', '/', '\\', ':', ';', "'", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '(', ')', '%']

raw_txt_file = open('NER_Dataset/train_bio.txt', 'r', encoding="utf8").readlines()  
file = open('NER_Dataset/train_bio_new.txt', "w", encoding="utf8") 

for line in raw_txt_file:  
    flag = 0
    for item in black_list:
        segment = line.split('\t')[0]
        if item in segment:
            flag = 1
            break
    if flag == 0:
        file.write(line )

file.close()