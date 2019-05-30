import sys 
import numpy as np 

# data loading
# raw_txt_file = open('NER_Dataset/train_bio.txt', 'r', encoding="utf8").readlines()  
raw_txt_file = open('NER_Dataset/test_bio.txt', 'r', encoding="utf8").readlines()  

sentence_arr = []
word_arr = []
counter = 0

for line in raw_txt_file:    
    counter += 1 
    stripped_line = line.strip().split('\t')
    word_arr.append(stripped_line) 
    
    # if line == "<S>		<S>\n" or line == "<S>		<S>":  
    if "<S>" in line:
        sentence_arr.append(word_arr[:-1])
        word_arr = [] 
        
# print (len(sentence_arr)) 
    
lengths = [len(x) for x in sentence_arr]
# print ('Input sequence length range: ', max(lengths), min(lengths))

X = [[c[0]  for c in x] for x in sentence_arr] 
y = [[c[3] for c in y] for y in sentence_arr] 
all_text = [c for x in X for c in x]

words    = list(set(all_text)) 
word2ind = {word: index for index, word in enumerate(words)} 
ind2word = {index: word for index, word in enumerate(words)}

# labels = list(set([c for x in y for c in x])) 
labels = ['B-TIME','I-TIME', 'B-LOCATION', 'I-LOCATION', 'B-DATE', 'I-DATE', 
          'B-PERSON', 'I-PERSON', 'B-MONEY', 'I-MONEY', 'B-ORGANIZATION',
          'I-ORGANIZATION', 'I-PERCENT',
          'B-PERCENT', 'O']
          
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}

# print ('Vocabulary size:', len(word2ind), len(label2ind))

# maxlen = max([len(x) for x in X])
maxlen = 184 # the maximum lenght from training set
# print ('Maximum sequence length:', maxlen)

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result
    
    
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## remove symbols from data
black_list = ['"', '.', ',', '/', '\\', ':', ';', "'", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '(', ')', '%']
# for row in X:
    # for word in row: 
        # for item in black_list:
            # if item in word:
                # print (word)
                # row.remove(word)
                
                # break
arr_i = []
arr_j = []


print ("the searching is started")

for i in range(len(X)-1, -1, -1):
    for j in range(len(X[i])-1, -1, -1): 
        for item in black_list:  
            if item in X[i][j]: 
                # print ("added => ", X[i][j])
                arr_i.append(i)
                arr_j.append(j)
                break

print ("the deleting is started")

file = open("test_bio_one_line.txt","w")  
for i in range(len(X)): 
    for j in range(len(X[i])): 
        flag = 0
        for k in range(len(arr_i)):
            if arr_i[k] == i and arr_j[k] == j:
                flag =1 
                break
        if flag == 0:
            file.write( X[i][j] + " " )  
    
 
file.close()    
    

# new_X = []
# for i in range(len(X)):
    # row_arr = []
    # for j in range(len(X[i])): 
        # flag = 0
        # for k in range(len(arr_i)):
            # if arr_i[k] == i and arr_j[k] == j:
                # flag =1 
                # break
        # if flag == 0:
            # row_arr.append(X[i][j])
    # new_X.append(row_arr)

# print ("writing into file")  

# X = new_X

# file = open("testfile.txt","w")  

# for row in X:
    # for word in row:
        # file.write( str(word) + " ")




