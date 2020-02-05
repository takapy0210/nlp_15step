import unicodedata
from os.path import dirname, join, normpath
import fire
import MeCab
import neologdn
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


class DialogueAgent:
    def __init__(self):
        self.tagger = MeCab.Tagger()

    def _tokenize(self, text):
        text = unicodedata.normalize('NFKC', text)
        text = neologdn.normalize(text)
        text = text.lower()

        node = self.tagger.parseToNode(text)
        result = []
        while node:
            features = node.feature.split(',')

            if features[0] != 'BOS/EOS':
                if features[0] not in ['助詞', '助動詞']:
                    token = features[6] \
                            if features[6] != '*' \
                            else node.surface
                    result.append(token)

            node = node.next

        return result

    def train(self, texts, labels, unit, epoch):
        vectorizer = TfidfVectorizer(tokenizer=self._tokenize, ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(texts)

        feature_dim = len(vectorizer.get_feature_names())
        n_labels = max(labels) + 1  # 49クラス分類

        mlp = Sequential()
        mlp.add(Dense(units=unit, input_dim=feature_dim, activation='relu'))
        mlp.add(Dense(units=n_labels, activation='softmax'))
        mlp.compile(loss='categorical_crossentropy', optimizer='adam')

        labels_onehot = to_categorical(labels, n_labels)
        mlp.fit(tfidf, labels_onehot, epochs=epoch)

        self.vectorizer = vectorizer
        self.mlp = mlp

    def predict(self, texts):
        tfidf = self.vectorizer.transform(texts)
        predictions = self.mlp.predict(tfidf)
        predicted_labels = np.argmax(predictions, axis=1)  # 一番確率の高いindexを取得
        return predicted_labels


def main(mode='agent', input_text='名前を教えてよ', unit=32, epoch=100) -> str:

    # スクリプト実行ディレクトリを絶対パスで取得する
    BASE_DIR = normpath(dirname(__file__))

    training_data = pd.read_csv(join(BASE_DIR, 'dialogue_agent_data/training_data.csv'))

    # 学習
    dialogue_agent = DialogueAgent()
    dialogue_agent.train(training_data['text'], training_data['label'], unit, epoch)

    if mode == 'agent':
        with open(join(BASE_DIR, 'dialogue_agent_data/replies.csv')) as f:
            replies = f.read().split('\n')

        # テキストに対する返答のクラスを予測し、対象の返答を出力する
        predictions = dialogue_agent.predict([input_text])
        predicted_class_id = predictions[0]

        print(replies[predicted_class_id])

    else:
        # Evaluation
        test_data = pd.read_csv(join(BASE_DIR, 'dialogue_agent_data/test_data.csv'))
        predictions = dialogue_agent.predict(test_data['text'])
        print(accuracy_score(test_data['label'], predictions))

    return 'Success!'


if __name__ == '__main__':

    fire.Fire(main)
