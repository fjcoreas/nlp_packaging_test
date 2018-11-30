import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
nlp = spacy.load("en_core_web_sm")

def lemmatize(doc):
    return[token.lemma_ for token in doc if not token.is_punct and not token.is_space and token.lower_ not in STOP_WORDS] #Lemmatize function to be used

#1. Write a function tf that receives a string and a spaCy Doc and returns the number of times the word appears in the lemmatized Doc
def tf(string,doc):
#obtaining lemmas of significant words in the given doc
    lemmas = lemmatize(doc)
#prints out the # of times the lemmatized string appears in the lemmatized doc
    return lemmas.count(''.join(lemmatize(nlp(string))))

#2. Write a function idf that receives a string and a list of spaCy Docs and returns the inverse of the number of docs that contain the word

def idf(string, list_docs):
# Used to determine the number of documents containing the string
  counter = 0
# Loop used to lemmatize and count string appearences per doc in a list of lemmatized docs
  for doc in list_docs:
    lemmas = lemmatize(doc)
# if used to count the lemmatized string in list of lemmatized docs, when count > 0 counter increases by 1
    if lemmas.count(''.join(lemmatize(nlp(string)))) > 0:
      counter += 1
# when counter is greater than 0 function divides 1 / (total number of docs in a list of docs containing the string)
  if counter > 0:
    return 1/counter
  else:
    return 0


def tf_idf(string, doc, list_docs):
    # multiplying the previously created functions to obtain tf*idf
    tfidf = idf(string, list_docs) * tf(string, doc)
    return (tfidf)

#4. Write a function all_lemmas that receives a list of Docs and returns a set of all available lemmas

def all_lemmas(list_docs):

# set of available lemmas in list of docs
  availemma = set()

# loop used to append each new lemma to availemma
  for doc in list_docs:
    availemma |= set(lemmatize(doc))
  return availemma

def tf_idf_doc(doc,list_docs):
# appending to each lemma from all_lemmas its td*idf value
  dictlemma = {lemma : tf_idf(lemma, doc, list_docs) for lemma in all_lemmas(list_docs)}
  return dictlemma

def tf_idf_scores(list_docs):
  documents = [tf_idf_doc(doc, list_docs) for doc in list_docs]  #returns dictionary list
  df = pd.DataFrame(documents) #converts list into dataframe
  return df