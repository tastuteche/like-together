from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
import pickle
import os
from os import path
from nltk import word_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
# from nltk.stem.lancaster import LancasterStemmer
# from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
# stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")
# lanste = LancasterStemmer()
import string
translator = str.maketrans('', '', string.punctuation)

from nltk.tokenize import wordpunct_tokenize

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')',
                   '[', ']', '{', '}'])  # remove it if you need punctuation


def tokenize_fast(text):
    return [stemmer.stem(i.lower()) for i in wordpunct_tokenize(text) if i.lower() not in stop_words]


def tokenize(text):
    return [stemmer.stem(i) for i in word_tokenize(text.lower().translate(translator)) if i not in stop]


def _clean_line(filename):
    with open(filename, "r") as f:
        return f.read()


def _get_files(dir_name):
    for filename in os.listdir(dir_name):
        print(filename)
        full_name = path.join(dir_name, filename)
        if path.isfile(full_name):
            text = _clean_line(full_name)
            if len(text) > 0:
                yield (filename, text)


def get_tfidf(dir_name):
    l = list(_get_files(dir_name))
    print(len(l))
    filename_list, text_list = zip(*l)
    files = numpy.array(text_list)

    doc_feat = TfidfVectorizer(
        tokenizer=tokenize, stop_words='english').fit_transform(files)

    pickle.dump(doc_feat, open("doc_feat.pickle", "wb"))
    pickle.dump(filename_list, open("filename_list.pickle", "wb"))


def load_tfidf():
    doc_feat = pickle.load(open("doc_feat.pickle", "rb"))
    filename_list = pickle.load(open("filename_list.pickle", "rb"))
    return doc_feat, filename_list
