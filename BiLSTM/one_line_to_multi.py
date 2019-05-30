import sys 
import numpy as np 
import random


input = open('NER_Dataset/test_bio_one_line.txt', 'r').readlines() 
file = open('NER_Dataset/test_multiline.txt', "w")  

counter = 0
a = random.randint(25,35) 
split = input[0].split(" ")


for i in range(len(split)):
    file.write(split[i] + " ")
    if counter == a:
        a = random.randint(25, 35)
        counter = 0
        file.write( "\n")
    counter += 1




file.close()