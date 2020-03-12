import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as lr

from tokenizer import tokenize

model = Word2Vec.load('./latest-ja-word2vec-gensim-model/word2vec.gensim.model')

def calc_text_feature(text):
    """
    単語の分散表現をもとにして、textの特徴量を求める。
    textをtokenizeし、各tokenの分散表現を求めたあと、
    すべての分散表現の合計をtextの特徴量とする。
    """
    tokens = tokenize(text)

    word_vectors = np.empty((0, model.wv.vector_size))
    for token in tokens:
        try:
            word_vector = model[token]
            word_vectors = np.vstack((word_vectors, word_vector))
        except KeyError:
            pass

    if word_vectors.shape[0] == 0:
        return np.zeros(model.wv.vector_size)
    return np.sum(word_vectors, axis=0)


# 評価
training_data = pd.read_csv('./dialogue_agent_data/training_data.csv')
test_data = pd.read_csv('./dialogue_agent_data/test_data.csv')

X_train = np.array([calc_text_feature(text) for text in training_data['text']])
y_train = np.array(training_data['label'])

X_test = np.array([calc_text_feature(text) for text in test_data['text']])
y_test = np.array(test_data['label'])

# model = SVC()
# model = RandomForestClassifier()
model = lr()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
