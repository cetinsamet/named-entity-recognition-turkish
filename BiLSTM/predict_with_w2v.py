import sys
import gensim
from keras.models import model_from_json 
import numpy as np
from gensim.models import KeyedVectors,  Word2Vec 
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split 
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


# model loading
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)  
loaded_model.load_weights('model_with_w2v.h5')   

# w2v_model = KeyedVectors.load_word2vec_format('w2v_model.bin')
w2v_model = gensim.models.Word2Vec.load("new\\dunya-nz_50.w2v")


# data loading
raw_txt_file = open('NER_Dataset/test_bio_new_lowercase.txt', 'r', encoding="utf8").readlines() 
# raw_txt_file = open('NER_Dataset/test2.txt', 'r', encoding="utf8").readlines() 

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

print ('Vocabulary size:', len(word2ind), len(label2ind))

# maxlen = max([len(x) for x in X])
maxlen = 121     # the maximum lenght from training set
print ('Maximum sequence length:', maxlen)

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result
    

X_enc = [[w2v_model[c] for c in x] for x in X] 

max_label = max(label2ind.values()) + 1
y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

X_enc = pad_sequences(X_enc, maxlen=maxlen)
X_enc = X_enc.reshape(-1, maxlen*100) 
y_enc = pad_sequences(y_enc, maxlen=maxlen) 

pred = loaded_model.predict_classes(X_enc)
print (pred.shape)
print (y_enc.shape)


##''''''''''''''''''''''''''''''''''''''''''''''''


nb_classes = 16
targets = pred.reshape(-1)
one_hot_pred = np.eye(nb_classes)[targets]

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
flat_X = [item for sublist in X for item in sublist]


print (flat_pred[:20])
# print (flat_y[:20])
print (flat_X[:20])

pred_idx = [label2ind[c] for c in flat_pred]
y_idx    = [label2ind[c] for c in flat_y]

# print (pred_idx[:100])
# print (y_idx[:100])




print (f1_score(flat_y, flat_pred) )
print (accuracy_score(flat_y, flat_pred) )
print (classification_report(flat_y, flat_pred))
a = input("here we are ....")



    
    
    
    
    
    
    
    