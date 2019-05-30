import io, sys , numpy as np
from gensim.models import KeyedVectors,  Word2Vec  
from keras import optimizers
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate,Reshape
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from keras import backend as K 


model = Word2Vec.load("w2v\\dunya-nz_50.w2v") 


def data_loader(data_path, def_maxlen = 0):
    raw_txt_file = open(data_path, 'r', encoding="utf8").readlines()

    sentence_arr = []
    word_arr = []
    counter = 0

    for line in raw_txt_file:  
        # print (line)
        counter += 1 
        stripped_line = line.strip().split('\t')
        word_arr.append(stripped_line) 
        
        if "<S>" in line:  
            sentence_arr.append(word_arr[:-1])
            # print (word_arr)
            word_arr = [] 
            
    # print (len(sentence_arr)) 
    
    lengths = [len(x) for x in sentence_arr]
    print ('Input sequence length range: ', max(lengths), min(lengths))   

    X        = [[c[0] for c in x] for x in sentence_arr] 
    y        = [[c[3] for c in y] for y in sentence_arr] 
    all_text = [c for x in X for c in x]

    words = list(set(all_text)) 
    word2ind = {word: index for index, word in enumerate(words)} 
    ind2word = {index: word for index, word in enumerate(words)}
    
    labels = ['B-TIME','I-TIME', 'B-LOCATION', 'I-LOCATION', 'B-DATE', 'I-DATE', 
              'B-PERSON', 'I-PERSON', 'B-MONEY', 'I-MONEY', 'B-ORGANIZATION',
              'I-ORGANIZATION', 'I-PERCENT',
              'B-PERCENT','O'] 


    label2ind = {label: (index + 1) for index, label in enumerate(labels)}
    print (label2ind)

    # sys.exit()
    ind2label = {(index + 1): label for index, label in enumerate(labels)}

    print ('Vocabulary size:', len(word2ind), len(label2ind)) 
    
    maxlen = max([len(x) for x in X]) if def_maxlen == 0 else def_maxlen
    print ('Maximum sequence length:', maxlen)

    def encode(x, n):
        result = np.zeros(n)
        result[x] = 1
        return result


    X_enc = [[model[c] for c in x] for x in X] 
    
    max_label = max(label2ind.values()) + 1 

    y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]        
    y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

    X_enc = pad_sequences(X_enc, maxlen=maxlen)
    y_enc = pad_sequences(y_enc, maxlen=maxlen) 
    return X_enc, y_enc, word2ind, label2ind, maxlen
    
#----------------------------------------------------------------------------

print ("train data is loading ...")
X_train, y_train, word2ind, label2ind, maxlen = data_loader('NER_Dataset/train_bio_new_lowercase.txt', 0)
X_train = X_train.reshape(-1, maxlen*100) 
print ("X_train",X_train.shape)
print ("y_train",y_train.shape)

max_features = len(word2ind)
print ("max_features", max_features)
out_size = len(label2ind) + 1
print ("test data is loading ...")
X_test, y_test, _, _, _ = data_loader('NER_Dataset/test_bio_new_lowercase.txt', maxlen)
X_test = X_test.reshape(-1, maxlen*100)

model = Sequential()     
model.add(Embedding(8, 8, input_length=maxlen*100, mask_zero=False))   
model.add(TimeDistributed(Dense(8, input_shape = (None, 100)))) 
model.add(Flatten())   
model.add(Dense(maxlen))  
model.add(Reshape((maxlen, 1)))   
model.add(Bidirectional(LSTM(32, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))  ) 
model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))   
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()


batch_size = 32
model.fit(X_train, y_train, batch_size=batch_size, epochs=15, validation_data=(X_test, y_test))
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("model_with_w2v.h5")
print("Saved model to disk")

