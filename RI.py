from nltk.corpus import stopwords
from math import sqrt, log10
import os
import codecs
import treetaggerwrapper

#*****************************************************************************************************************
#*** Vector class 
class Vector:    
    
    #+++ Contractors
    def __init__(self, elements):        
        self.elements = elements
        self.dim = len(elements)

    #+++ Print vector
    def Print(self):
        print(self.elements)

    #+++ Add operator
    def __add__(self, other):       
        if self.dim == other.dim:            
            elements = list(x+y for x, y in zip(self.elements, other.elements))
            return Vector(elements)

    #+++ Sub operator
    def __sub__(self, other):       
        if self.dim == other.dim:            
            elements = list(x-y for x, y in zip(self.elements, other.elements))
            return Vector(elements)

    #+++ Multiplication operator
    def __mul__(self, other):
        if self.dim == other.dim:            
            cosinus = sum(x*y for x, y in zip(self.elements, other.elements))
            return cosinus
             
    #+++ Multiplication operator
    def __rmul__(self, other):
        if type(other) in (int, float):
            elements = list(element*other for element in self.elements)
            return Vector(elements)
    
    #+++ normalized vector
    def Normal(self):        
        sum_square = sqrt(sum (element**2 for element in self.elements))
        elements = list(element/sum_square for element in self.elements)        
        return Vector(elements)

#*****************************************************************************************************************
#*** PyTreeTagger class
class PyTreeTagger:    
    #Returns a list of lemma distinct and a list of sentences which contain the tokens lemmatized.
    @staticmethod    
    def Get_Sent_Lemma(text, use_stopword = 1, lang='fr'):
        # Pre-process the corpus using TreeTagger        
        list_stopword = set(['.','!',',','?','\\','/','(',')',';',':'])
        # remove the stopwords
        if use_stopword == 0:
            dic_lang = {'en':'english', 'fr':'french', 'es': 'spanish'}
            list_stopword = list_stopword | set(stopwords.words(dic_lang[lang]))
        
        tagger = treetaggerwrapper.TreeTagger(TAGLANG=lang)    
        # Replace the characters '’' and '.' of the text for avoid the errors in the TreeTagger
        text = text.replace('’','\'').replace('.','. ').replace('\ufeff','')
        # Process each sentence using TreeTagger.
        tags = tagger.tag_text(text)                    
        sent_lemma =[x.split('\t')[2] for x in tags if x.split('\t')[2] not in list_stopword]                 

        return sent_lemma
            
#*****************************************************************************************************************
#*** Random indexing training class
class RI_Training:

    #+++ Contractor
    def __init__(self, path_corpus, file_vector, dim, lang='fr'):
        # Open the corpus training 
        f = codecs.open(path_corpus, encoding='utf-8')
        lines = f.readlines()
        f.close()        
        #
        sentences =[]
        terms =set([])                 
        for line in lines:                                          
            sent_lemma = PyTreeTagger.Get_Sent_Lemma(line)
            #--- Add the sentences and the words into list after lemmatise them
            sentences.append(sent_lemma)
            terms = terms|set(sent_lemma)        
        #--- Set value for the attributes of class RI_Training
        self.terms = list(terms)
        self.terms.sort()
        self.dim = dim
        self.sentences = sentences
        #------------------------------------------------------------------------------
        # Generate vectors index for each term in the corpus               
        f = open(file_vector,'r')        
        lines = f.readlines()
        f.close()
        # Get number of coordinate non null
        nb_half_coor = int(len(lines[0].split('-'))/2)
        # Get number of term
        n = len(self.terms)
        # initialize dictionar of vector index under form [term -> vector index]
        dic_vector_index ={}
        #
        for i in range(0,n):
            coordinates = lines[i].split('-')
            elements =[0]*self.dim            
            # Assign elements of vectors index for the terms in the list
            for j in range(0,int(nb_half_coor)):
                elements[int(coordinates[j])] = 1
                elements[int(coordinates[nb_half_coor + j])] = -1                
            # Put pair [term-vector index] into dictionary
            dic_vector_index[self.terms[i]] = Vector(elements)                       
        # Set value for the dic_vector_index attribute of class RI_Training
        self.dic_vector_index = dic_vector_index

           
    #+++ Calculate the sentence frequency for each term distinct in corpus training
    def Calculate_Isf(self):
        nb_sentence = len(self.sentences)
        dic_sf ={}
        for sentence in self.sentences:
            # Get a list of term distinct       
            distinct_terms = []
            for term in sentence:
                if term not in distinct_terms: 
                    distinct_terms.append(term)                    
            # Count number of sentence which contain the term considered 
            for term in distinct_terms:
                if term in dic_sf:
                    dic_sf[term]+= 1
                else:
                    dic_sf[term] = 1
        # 
        dic_isf = {}
        for key, value in dic_sf.items():
            dic_isf[key] = log10((nb_sentence+1)/(value+1));
            #print(key + " -> " + str(dic_isf[key]))
        return dic_isf ;
    
    #+++ Acummulate the contexts for a term in one sentence
    def Context_Term_In_Sentence(self, terms_of_sentence, nb_term, pos, window_width, dic_isf):   

        vector_context = Vector([0]*self.dim)        
        # Accumulate the rights contexts
        if pos< nb_term-1:
            r_weight = 1.0
            window = 0
            for i in range(pos+1, nb_term):
                window+=1
                vector_context = vector_context + r_weight * dic_isf[terms_of_sentence[i]] * self.dic_vector_index[terms_of_sentence[i]]
                r_weight = r_weight * 0.5
                if window == window_width: break
                
        # Accumulate the lefts contexts
        if pos > 0:
            l_weight =1.0
            window =0
            for i in range(pos-1,-1,-1):
                window+=1
                vector_context = vector_context + l_weight * dic_isf[terms_of_sentence[i]] * self.dic_vector_index[terms_of_sentence[i]]
                l_weight = l_weight * 0.5
                if window == window_width: break      
        return vector_context
    
    #+++ Accumulate all contexts for each term in the corpus training
    def Accummulate_Context_Term_In_Corpus(self, window):       
        dic_isf = self.Calculate_Isf()
        dic_semantic_term ={}
        #
        for terms in self.sentences:            
            nb_term = len(terms)
            for i in range(0, nb_term):
                if terms[i] in dic_semantic_term:
                    dic_semantic_term[terms[i]] +=  self.Context_Term_In_Sentence(terms,nb_term , i, window , dic_isf)                    
                else:
                    dic_semantic_term[terms[i]] = self.Context_Term_In_Sentence(terms, nb_term, i, window, dic_isf)            

        #
        #Calculate vector centroid
        k = ''
        for key in dic_semantic_term:
            k = key
            break
        centroid = Vector([0]*dic_semantic_term[k].dim)
        for key in dic_semantic_term:
            centroid += dic_semantic_term[key]

        centroid = (1/len(dic_semantic_term))*centroid
        #
        for key in dic_semantic_term:
            dic_semantic_term[key] = dic_semantic_term[key] - centroid
        return dic_semantic_term

#*****************************************************************************************************************  
#*** Similarity class
class Similarity:

    #+++ Contractor
    def __init__(self, dic_semantic_term, lang = 'fr'):        
        self.dic_semantic_term = dic_semantic_term
        self.lang = lang
        

    #+++ Calculate similarity between tow sentences using the sum of the vector
    def Sim_Sentence(self, sent1, sent2):

        terms_sent1 = PyTreeTagger.Get_Sent_Lemma(sent1, 1, self.lang)

        v1= Vector([0]*self.dic_semantic_term[terms_sent1[0]].dim)
        for term in terms_sent1:
            v1+= self.dic_semantic_term[term]
        v1 = (1/len(terms_sent1))*v1
        
        terms_sent2 = PyTreeTagger.Get_Sent_Lemma(sent2, 1, self.lang)
        v2= Vector([0]*self.dic_semantic_term[terms_sent2[0]].dim)
        for term in terms_sent2:
            v2+= self.dic_semantic_term[term]
        v2 = (1/len(terms_sent2))*v2
        
        return v1.Normal() * v2.Normal()

    #+++ Calculate similarity between tow sentences using each vector lexical in the sentences   
    def Sim_Lexical(self, sent1, sent2):
        terms_sent1 = PyTreeTagger.Get_Sent_Lemma(sent1, self.lang)            
        terms_sent2 = PyTreeTagger.Get_Sent_Lemma(sent2, self.lang)
        #
        sum_max_sim_lex1 = 0.0
        for ts1 in terms_sent1:
            v = self.dic_semantic_term[ts1].Normal()
            sum_max_sim_lex1 += max(v * self.dic_semantic_term[x].Normal() for x in terms_sent2)            
        sum_max_sim_lex1 = sum_max_sim_lex1/len(terms_sent1)
        #
        sum_max_sim_lex2 = 0.0
        for ts2 in terms_sent2:
            v = self.dic_semantic_term[ts2].Normal()
            sum_max_sim_lex2 += max(v * self.dic_semantic_term[x].Normal() for x in terms_sent1)
        sum_max_sim_lex2 = sum_max_sim_lex2/len(terms_sent2)
          
        return (sum_max_sim_lex1 + sum_max_sim_lex2)/2

#*****************************************************************************************************************  

#Get data from MongoDB
from pymongo import MongoClient
client = MongoClient('localhost',27017)
db = client.RI_FR
cursor = db.dic_semantic_term.find()

#print(db.dic_semantic_term.find({'key':"Dieu"})[0])

dic ={}
for doc in cursor:
    dic[doc['key']] = Vector(doc['vec'])

#Calculate sim
ri = Similarity(dic)
print(str(ri.Sim_Lexical("fais le bien si tu veux qu'on te le fasse","fais le bien si tu veux recevoir du bien")))
print(str(ri.Sim_Lexical("fais le bien si tu veux qu'on te le fasse","la mort est la fauchaison de l'espoir")))












'''
f = codecs.open("D:\Test\ESB_226_Litteral.txt", encoding='utf-8')
lines = f.readlines()
f.close()
pairs_esb = list([line.split('\t')[0],line.split('\t')[1]] for line in lines if len(line)>10)
'''





'''
b = codecs.open("D:\\Test\\sim_ri_Litteral.txt", "w")
for esbs in pairs_esb:
    b.write(str(ri.Sim_Lexical(esbs[0],esbs[1])) +'\n')
b.close()
'''






'''
f0 = codecs.open("D:\\Test\\gs.a234")
lines = f0.readlines()
f0.close()

a = codecs.open("D:\\Test\\gs234.txt", "w")
for line in lines:
    if len(line.replace('\r\n','').replace('\n',''))>0:
        a.write(line.replace('\r\n','').replace(',','.').replace('\n','')+'\n')
a.close()
'''







'''
fichiers = ["D:\Test\ESB_226_Litteral.txt","D:\Test\ESB_226_Figure.txt","D:\Test\ESB_226_Lecons.txt"]

for fichier in fichiers:
    f = codecs.open(fichier, encoding='utf-8')
    lines = f.readlines()
    f.close()
    pairs_esb = list([line.split('\t')[0],line.split('\t')[1]] for line in lines if len(line)>10)

    f1 = codecs.open(fichier+"_lemma.txt", "w", "utf-8")

    for pair in pairs_esb:
        lemmas1 = PyTreeTagger.Get_Sent_Lemma(pair[0])
        sent_lemma1 =''
        for lemma in lemmas1: 
            sent_lemma1 += lemma + ' '
    
        lemmas2 = PyTreeTagger.Get_Sent_Lemma(pair[1])
        sent_lemma2 =''
        for lemma in lemmas2: 
            sent_lemma2 += lemma + ' '

        f1.write(sent_lemma1 + '\t' + sent_lemma2 + '\n')
    
    f1.close()

'''
