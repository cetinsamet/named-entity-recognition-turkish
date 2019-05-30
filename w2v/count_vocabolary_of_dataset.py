import io, sys , numpy as np

  
#----------------------------------------------------------------------------


raw_txt_file = open("dunya-nz_new.txt", 'r', encoding="utf8").readlines()

voc_set = set()

counter = 0
for line in raw_txt_file:  
    split = line.split(" ")
    for item in split:
        counter += 1
        voc_set.add(item)
        
        
print ("Vocabolary size: ", len(voc_set))
print ("all words: ", counter)