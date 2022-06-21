import nltk
# nltk.download("punkt")
import numpy as np

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_word(tokenized_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
 # stem each word
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # initializing the bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence: 
            bag[idx] = 1.0

    return bag 



# # tokenize testing
# a = "Hi How are you?"
# print(a)
# a = tokenize(a)
# print(a)

# # stem testing
# words = ["Oraganize", "oraganization", "Oraganizes", "Oraganizing"]
# stemmed_word = [stem(w) for w in words]
# print(stemmed_word)

# # to check the word is in bag or not
# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bog = bag_of_word(sentence, words)
# print(bog)