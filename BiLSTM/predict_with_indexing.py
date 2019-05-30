import sys
from keras.models import model_from_json 
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# model loading
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)  
loaded_model.load_weights('model_with_indexing.h5')   


# data loading
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
        
print (len(sentence_arr)) 
    
lengths = [len(x) for x in sentence_arr]
print ('Input sequence length range: ', max(lengths), min(lengths))

X = [[c[0] for c in x] for x in sentence_arr] 
y = [[c[3] for c in y] for y in sentence_arr] 
all_text = [c for x in X for c in x]

words = list(set(all_text)) 
word2ind = {word: index for index, word in enumerate(words)} 
ind2word = {index: word for index, word in enumerate(words)}

# labels = list(set([c for x in y for c in x])) 
labels = ['B-TIME','I-TIME', 'B-LOCATION', 'I-LOCATION', 'B-DATE', 'I-DATE', 
          'B-PERSON', 'I-PERSON', 'B-MONEY', 'I-MONEY', 'B-ORGANIZATION',
          'I-ORGANIZATION', 'I-PERCENT',
          'B-PERCENT', 'O']
          
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}

print ('Vocabulary size:', len(word2ind), len(label2ind))

# maxlen = max([len(x) for x in X])
maxlen = 184 # the maximum lenght from training set
print ('Maximum sequence length:', maxlen)

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

X_enc = [[word2ind[c] for c in x] for x in X]

max_label = max(label2ind.values()) + 1
y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

X_enc = pad_sequences(X_enc, maxlen=maxlen)
# print (y_enc[0])
y_enc = pad_sequences(y_enc, maxlen=maxlen)
# print (y_enc[0])
# sys.exit(0)

pred = loaded_model.predict_classes(X_enc)
print (pred.shape)
print (y_enc.shape)


##''''''''''''''''''''''''''''''''''''''''''''''''


nb_classes = 16
targets = pred.reshape(-1)
one_hot_pred = np.eye(nb_classes)[targets]

# print (one_hot_pred)
# print (one_hot_pred.shape)

# print ("=================================")

# print (y_enc.reshape(-1, nb_classes))
# print (y_enc.shape)

print ('---------------------------------') 


depad_pred = []
for i in range(len(X)): 
    t = y[i] 
    
    aa = [] 
    cur = pred[i][len(pred[i])-len(y[i]):]
    for c in cur:
        if c == 0: 
            aa.append('O')
        else:
            aa.append(ind2label[c])
        
    # print (aa) 
    depad_pred.append(aa)
    
    
    # _ = input("Type something to test this out: ")
    
print ("============================================================")
flat_pred = [item for sublist in depad_pred for item in sublist]
flat_y = [item for sublist in y for item in sublist]

# print (flat_pred[:100])
# print (flat_y[:100])

pred_idx = [label2ind[c] for c in flat_pred]
y_idx    = [label2ind[c] for c in flat_y]

# print (pred_idx[:10])
# print (y_idx[:10])

print("f1_score",        f1_score       (y_idx, pred_idx, average="macro"))
print("precision_score", precision_score(y_idx, pred_idx, average="macro"))
print("recall_score",    recall_score   (y_idx, pred_idx, average="macro")) 

con = confusion_matrix(y_idx, pred_idx) 
print (con)


r, c = con.shape
precision = []
recall = []
for i in range(r): 
    tp = con[i][i]
    fp = np.sum(con[i, :]) - con[i][i]
    fn = np.sum(con[:, i]) - con[i][i]    
    tn_up = np.sum(con[0:i-1, 0:i-1]) if (i-1) >= 0 else 0
    tn_down = np.sum(con[i+1:r, i+1:c]) if (i+1) <= r else 0
    tn = tn_up + tn_down
    
    precision.append(tp/(tp+fp))
    recall.append(tp/(tp+fn))

print (precision)
print (recall)
print ("precision:", sum(precision) / 15)
print ("recall", sum(recall) /15)
    
    
    
    
    
    
    
    
    
    