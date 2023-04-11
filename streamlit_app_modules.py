# cleaning functions
from nltk.corpus import wordnet as wn
import contractions
import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk import pos_tag

def is_flat(lst):
    for item in lst:
        if isinstance(item, list):
            return False
    return True

def flatten(l):
    return [item for sublist in l for item in sublist]


def Negation(sentence):	
  '''
  Input: Tokenized sentence (List of words)
  Output: Tokenized sentence with negation handled (List of words)
  Source/Reference: https://towardsdatascience.com/increasing-accuracy-of-sentiment-classification-using-negation-handling-9ed6dca91f53
  Note: I changed the code a little to work with my cleaned and tokenized column, added function check_negation()
        It will now perform slightly better especially with whitespace tokenizer that I've been using ['I', 'dislike', 'this', 'product'] vs ['I', 'do', 'dislike', 'this', 'product']
        also added if syns != None: as sometimes, no synonyms are found for a word
  '''

  def check_negation(word):
    return any(item in word for item in ['not', "n't"])

  temp = int(0)
  for i in range(len(sentence)):
    # if the word before this word has a substring of 'not' or "n't"
      if check_negation(sentence[i-1]): # cuz of this line
          antonyms = []
          try:
            for syn in wn.synsets(sentence[i]):
                syns = wn.synsets(sentence[i])
                w1 = syns[0].name() # sometimes no synonyms found
                temp = 0
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
                max_dissimilarity = 0
                for ant in antonyms:
                    syns = wn.synsets(ant)
                    w2 = syns[0].name()
                    syns = wn.synsets(sentence[i])
                    w1 = syns[0].name() # sometimes no synonyms found
                    word1 = wn.synset(w1)
                    word2 = wn.synset(w2)
                    if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                        temp = 1 - word1.wup_similarity(word2)
                    if temp>max_dissimilarity:
                        max_dissimilarity = temp
                        antonym_max = ant
                        sentence[i] = antonym_max
                        sentence[i-1] = ''
          except Exception as e:
            print("Error, likely no synonym/antonym found for", sentence[i], " :", e)
            pass
  while '' in sentence:
      sentence.remove('')
  return sentence

def decontracted(phrase):
    return contractions.fix(phrase)

whitespace_tokenizer = RegexpTokenizer(r"(?<=\b)[a-zA-Z'’]+(?=\b)|\b[a-zA-Z]+\b")
#     (?<=\b)[a-zA-Z']+(?=\b) : Matches any word that contains an apostrophe preceded and followed by a word boundary
    # \b[a-zA-Z]+\b : Matches any word that does not contain whitespaces or punctuation.

def preprocess(text):
    sents = sent_tokenize(text)
    # further split if necessary, realised there are some sentences ending with . but no space after, which is not recognised by libraries :")
    # this wrongful split causes issues later with weird words formed after removing punctuation :")
    sents = [re.sub(r'([a-z])\.([A-Z])', r'\1.\2', w1).split('.') for w1 in sents]
    sents = [item for sublist in sents for item in sublist if item != '']

    words = [whitespace_tokenizer.tokenize(x) for x in sents] # seaparates couldn't to couldn 't

    # handling negation before expanding contractions and removing stopwords
    # doesnt work with ’ so i replace it to ' first
    words = [[x.replace('’', "'") for x in sublist] for sublist in words]
    words = [Negation(x) for x in words]
    # is working: This way we don’t think about it, but it still has the power to affect the way we act and view things. to [This, way, we, forget, about, it, but, it, still, has, the, power, to, affect, the, way, we, act, and, view, things]

    # expand contractions before removing punctuation, add words back into list
    decontracted_sentences = []
    for i in range(len(words)):
        words[i] = [decontracted(x).split(" ") for x in words[i]]
        words[i] = [item for sublist in words[i] for item in sublist if item != '']
        decontracted_sentences.append(words[i])

    # remove puncutation if not part of a word
    stripped = [x for x in decontracted_sentences if x not in list(string.punctuation)]
    
    return stripped

def addPOSTags(sentences):
    return [pos_tag(sent) for sent in sentences]

def cleanTokens(tokenList):
    stop_words = list(stopwords.words('english'))
    cleanedTokens = []
    for token, tag in tokenList:
        # remove punctuation and convert case
        token = token.translate(str.maketrans('', '', string.punctuation)).lower()
        # ignore words that are not alphabetic and are stopwords
        if token.isalpha():
            if token not in stop_words:
                cleanedTokens.append((token,tag))
    return cleanedTokens

# POS tagging
# import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# tag converter
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# lemmatize with pos tagging
def lemmatize(input_word, tag):
    wordnetTag = get_wordnet_pos(tag)
    lem = WordNetLemmatizer()
    return (lem.lemmatize(input_word, wordnetTag).lower(), tag)

# using model

def clean_text(text):
    # preprocess title column
    preprocessed_text = text
    preprocessed_text = preprocess(text)
    preprocessed_text = addPOSTags(preprocessed_text)
    preprocessed_text = [cleanTokens(sent) for sent in preprocessed_text]
    preprocessed_text = [[lemmatize(word, tag) for word, tag in sent] for sent in preprocessed_text]

    return preprocessed_text

# load model
import pickle

with open('models/12_topics_LDATunedModel.pkl', 'rb') as file:
    lda_model = pickle.load(file)

with open('models/lr_ovr_nouns_tuned_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)