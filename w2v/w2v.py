import gensim
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import decomposition
import time
s = time.time()

# train 
print ("Program is started")
sentences = gensim.models.word2vec.LineSentence("dunya-nz_new.txt", max_sentence_length=10000)
print ("data is loaded") 
size = 100 
window_size = 10 
epochs =  25
min_count = 0 
workers = 4  
print ("train word2vec model using gensim")
model = gensim.models.Word2Vec(sentences, sg=1, window=window_size, size=size, min_count=min_count, workers=workers, iter=epochs, sample=0.01)
print ("the model is build vocab.")    
model.train(sentences=sentences, epochs=25, total_examples=model.corpus_count) 
model.save("dunya-nz_50.w2v") 

# e = time.time()
# print ("exec time:", e-s)

# model = gensim.models.Word2Vec.load("dunya-nz_50.w2v")




liste = ['kadın','macera','korku','ayşe','murat']
for l in liste:
    try:
        print("\n{} : ".format(l), model.most_similar(l,topn=10))
    except Exception as e:
        print("\nHata: {}".format(e))

def plotWords(w2v):
    w2v_np = []
    labels = []
    for word in w2v.wv.vocab.keys():
        w2v_np.append(w2v[word])
        labels.append(word)
    print("Shape = {}".format(np.shape(w2v_np)))

    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(w2v_np)
    reduced= pca.transform(w2v_np)

    cnt=0
    lim =2.0 
    plt.rcParams.update({'font.size':9})
    plt.rcParams["figure.figsize"] = [12.0,12.0]
    for index, vec in enumerate(reduced):
        # if index < 50000: continue
        if cnt < 250:
            x,y=vec[0],vec[1]
            if x>lim or y>lim or x<-lim or y<-lim: continue
            plt.scatter(x,y)
            plt.annotate(labels[index],xy=(x,y))
            cnt+=1
    plt.show()
    
#model = gensim.models.Word2Vec.load("../input/dunya-nz.w2v")
plotWords(model)