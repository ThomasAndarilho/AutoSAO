# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:43:44 2023

@author: UQBU
"""

import pandas as pd
import numpy as np

# Leitura de dados originais
path = r"dados.csv"
df = pd.read_csv(path, sep= "|")

# Classificando tudo que não for "Conforme" para a classe 1
df['nc'] = np.where(df["Gravidade"] != "Conforme", 1, 0)

# Conversão do texto para minúsculo
df["texto"] = df["TELO_TX_LINHA"].str.lower()

# Elimnando as linhas sem texto longo para análise
df.dropna(subset = "TELO_TX_LINHA", inplace=True)
df["texto"]

# Pré-processamento com base em https://practicaldatascience.co.uk/machine-learning/how-to-preprocess-text-for-nlp-in-four-easy-steps
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
+nltk.download("machado");

def tokenize(column):
    """Tokenizes a Pandas dataframe column and returns a list of tokens.
    Args:
        column: Pandas dataframe column (i.e. df['text']).
    Returns:
        tokens (list): Tokenized list, i.e. [Donald, Trump, tweets]
    """
    tokens = nltk.word_tokenize(column)
    return [w for w in tokens if w.isalpha()] 

df["tokenized"] = df.apply(lambda x: tokenize(x["texto"]), axis=1)

# Removendo stopwords
nltk.download('stopwords');
def remove_stopwords(tokenized_column):
    """Return a list of tokens with English stopwords removed. 
    Args:
        column: Pandas dataframe column of tokenized data from tokenize()
    Returns:
        tokens (list): Tokenized list with stopwords removed.
    """
    stops = set(stopwords.words("english"))
    return [word for word in tokenized_column if not word in stops]
df["stopwords_removed"] = df.apply(lambda x: remove_stopwords(x['tokenized']), axis=1)
#df.head()

# Stemming
def apply_stemming(tokenized_column):
    """Return a list of tokens with Porter stemming applied.
    Args:
        column: Pandas dataframe column of tokenized data with stopwords removed.
    Returns:
        tokens (list): Tokenized list with words Porter stemmed.
    """

    stemmer = PorterStemmer() 
    return [stemmer.stem(word) for word in tokenized_column]
df["porter_stemmed"] = df.apply(lambda x: apply_stemming(x['stopwords_removed']), axis=1)

#Textos agrupados novamente
def rejoin_words(tokenized_column):
    """Rejoins a tokenized word list into a single string. 
    Args:
        tokenized_column (list): Tokenized column of words. 
    Returns:
        string: Single string of untokenized words. 
    """
    return ( " ".join(tokenized_column))
df["rejoined"] = df.apply(lambda x: rejoin_words(x['porter_stemmed']), axis=1)
#df.head()

# Definido o índice dos dados como o número da ordem de manutenção
df.set_index("NumOrdem", inplace = True)

# Realizada a divisão entre conjuntos de treino e de teste
from sklearn.model_selection import train_test_split
X_train_txt, X_test_txt, y_train, y_test = train_test_split(df.rejoined, df["nc"], test_size = 0.2, random_state = 0, stratify = df["nc"])

# Na próxima etapa é realizada a construção do vetorizador TF-IDF apenas com os dados de treino de forma que não haja vazamento de dados e um favorecimento injusto do TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_txt)
#X_train

# Em seguida é realizada a vetorização dos dados de teste com o modelo treinado nos dados de treino
X_test = vectorizer.transform(X_test_txt)
# X_test

# Redução de dimensionalidade para 100 usando conforme https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 1000, random_state = 0)
svd.fit(X_train)
X_train_new = svd.transform(X_train)
# Transforma em DataFrame e devolve o índice
X_train_new = pd.DataFrame(X_train_new, index = y_train.index)
#X_train_new

# Usado modelo treinado nos dados de treino para reduzir a dimensionalidade dos dados de teste
X_test_new = svd.transform(X_test)
# Transforma em DataFrame e devolve o índice
X_test_new = pd.DataFrame(X_test_new, index = y_test.index)
#X_test_new

X_train_new.to_excel("TF-IDF_X_train.xlsx")
X_test_new.to_excel("TF-IDF_X_test.xlsx")
y_train.to_excel("TF-IDF_y_train.xlsx")
y_test.to_excel("TF-IDF_y_test.xlsx")


