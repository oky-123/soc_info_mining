import re
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 整形メソッド
def preprocessing_text(text):
  en_stop = nltk.corpus.stopwords.words('english')

  def cleaning_text(text):
    # @の削除
    pattern1 = '@|%'
    text = re.sub(pattern1, '', text)
    pattern2 = '\[[0-9 ]*\]'
    text = re.sub(pattern2, '', text)
    # <b>タグの削除
    pattern3 = '\([a-z ]*\)'
    text = re.sub(pattern3, '', text)
    pattern4 = '[0-9]'
    text = re.sub(pattern4, '', text)
    return text

  def tokenize_text(text):
    text = re.sub('[.,]', '', text)
    return text.split()

  def lemmatize_word(word):
    # make words lower  example: Python =>python
    word=word.lower()

    # lemmatize  example: cooked=>cook
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
      return lemma

  def remove_stopwords(word, stopwordset):
    if word in stopwordset:
      return None
    else:
      return word

  text = cleaning_text(text)
  tokens = tokenize_text(text)
  tokens = [lemmatize_word(word) for word in tokens]
  tokens = [remove_stopwords(word, en_stop) for word in tokens]
  tokens = [word for word in tokens if word is not None]
  return tokens

def bow_vectorizer(docs):
  word2id = {}
  for doc in docs:
    for w in doc:
      if w not in word2id:
        word2id[w] = len(word2id)

  result_list = []
  for doc in docs:
    doc_vec = [0] * len(word2id)
    for w in doc:
      doc_vec[word2id[w]] += 1
    result_list.append(doc_vec)
  return result_list, word2id

def tfidf_vectorizer(docs):
  def tf(word2id, doc):
    term_counts = np.zeros(len(word2id))
    for term in word2id.keys():
      term_counts[word2id[term]] = doc.count(term)
    tf_values = list(map(lambda x: x/sum(term_counts), term_counts))
    return tf_values

  def idf(word2id, docs):
    idf = np.zeros(len(word2id))
    for term in word2id.keys():
      idf[word2id[term]] = np.log(len(docs) / sum([bool(term in doc) for doc in docs]))
    return idf

  word2id = {}
  for doc in docs:
    for w in doc:
      if w not in word2id:
        word2id[w] = len(word2id)

  return [[_tf*_idf for _tf, _idf in zip(tf(word2id, doc), idf(word2id, docs))] for doc in docs], word2id

# 0. 文章集合
docs = [
'View update is an important mechanism that allows updates on a view by translating them into the corresponding updates on the base relations.',
'The existing literature has shown the ambiguity of translating view updates.',
'To address this ambiguity, we propose a robust language-based approach for making view update strategies programmable and validatable.',
'Specifically, we introduce a novel approach to use Datalog to describe these update strategies.',
'We present a fragment of the Datalog language for which our validation is both sound and complete.',
'We propose a validation algorithm to check the well-behavedness of the written Datalog programs.',
'This fragment not only has good properties in theory but is also useful for solving practical view updates.',
'Furthermore, we develop an algorithm for optimizing user-written programs to efficiently implement updatable views in relational database management systems.',
'We have implemented our proposed approach.',
'The experimental results show that our framework is feasible and efficient in practice.'
]

# 1. 文章をデータ構造に変換
## 1.1. BoW
pp_docs = [preprocessing_text(text) for text in docs]
bow_vec, word2id = bow_vectorizer(pp_docs)
## 1.2. tf-idf
tfidf_vec, word2id = tfidf_vectorizer(pp_docs)

# 2. 距離尺度
## 2.1. cosine類似度
def calc_cosine(vector, vector_list):
  def cosine_similarity(list_a, list_b):
    inner_prod = np.array(list_a).dot(np.array(list_b))
    norm_a = np.linalg.norm(list_a)
    norm_b = np.linalg.norm(list_b)
    try:
      return inner_prod / (norm_a*norm_b)
    except ZeroDivisionError:
      return 1.0
  result = {}
  for i, x in enumerate(vector_list):
    result[i] = cosine_similarity(vector, vector_list[i])
  return result

### BoWに対してcosine類似度の計算
cosine_similarity_bows = []
for i in range(10):
    cosine_similarity_bow = calc_cosine(bow_vec[i], bow_vec[0:10])
    cosine_similarity_bows.append(list(cosine_similarity_bow.values()))

### tf-idfに対してcosine類似度の計算
cosine_similarity_tfidfs = []
for i in range(10):
    cosine_similarity_tfidf = calc_cosine(tfidf_vec[i], tfidf_vec[0:10])
    cosine_similarity_tfidfs.append(list(cosine_similarity_tfidf.values()))

## 2.2 ユークリッド距離
def euclidean_distance(list_a, list_b):
    diff_vec = np.array(list_a) - np.array(list_b)
    return np.linalg.norm(diff_vec, ord=2)

### BoWに対してユークリッド距離の計算
euclidean_distance_bows = [[0] * 10 for i in range(10)]
for i in range(10):
    for j in range(10):
        euclidean_distance_bows[i][j] = euclidean_distance(bow_vec[i], bow_vec[j])

### tf-idfに対してユークリッド距離の計算
euclidean_distance_tfidfs = [[0] * 10 for i in range(10)]
for i in range(10):
    for j in range(10):
        euclidean_distance_tfidfs[i][j] = euclidean_distance(tfidf_vec[i], tfidf_vec[j])

## BoWとtf-idfの結果図示
### 文書8
w = 0.8
plt.figure()
bow_8_9 = np.array(bow_vec[8]) + np.array(bow_vec[9])
bow_8_9_index = list(map(lambda t: t[0], filter(lambda t: t[1] > 0, enumerate(bow_8_9))))
print(bow_8_9_index)

bow_vec_8 = list(map(lambda t: t[1], (filter(lambda t: t[0] in bow_8_9_index, enumerate(bow_vec[8])))))
bow_vec_9 = list(map(lambda t: t[1], (filter(lambda t: t[0] in bow_8_9_index, enumerate(bow_vec[9])))))
tfidf_vec_8 = list(map(lambda t: t[1], (filter(lambda t: t[0] in bow_8_9_index, enumerate(tfidf_vec[8])))))
tfidf_vec_9 = list(map(lambda t: t[1], (filter(lambda t: t[0] in bow_8_9_index, enumerate(tfidf_vec[9])))))
bow_8_9_index = list(map(lambda x: str(x), bow_8_9_index))

# その単語の出現するドキュメントの個数
freqs = [0] * len(bow_vec[0])
for i in bow_8_9_index:
    i = int(i)
    for bow in bow_vec:
        if bow[i] > 0:
            freqs[i] += 1

print(freqs)
print(bow_vec[8])
print(bow_vec[9])

plt.bar(bow_8_9_index, bow_vec_8, align="center", width=w, label='BoW')
plt.bar(bow_8_9_index, tfidf_vec_8, align="center", width=w, label='tf-idf')
plt.title("BoW and tf-idf of document 9")
plt.xlabel("ID")
plt.grid(True)
plt.savefig('doc_8')
plt.close('all')

### 文書9
plt.bar(bow_8_9_index, bow_vec_9, align="center", width=w, label='BoW')
plt.bar(bow_8_9_index, tfidf_vec_9, align="center", width=w, label='tf-idf')
plt.title("BoW and tf-idf of document 10")
plt.xlabel("ID")
plt.grid(True)
plt.savefig('doc_9')
plt.close('all')

## 距離尺度によるヒートマップ
plt.figure()
sns.heatmap(cosine_similarity_bows)
plt.savefig('cosine_similarity_bows.png')
plt.close('all')

plt.figure()
sns.heatmap(cosine_similarity_tfidfs)
plt.savefig('cosine_similarity_tfidfs.png')
plt.close('all')

plt.figure()
sns.heatmap(euclidean_distance_bows)
plt.savefig('euclidean_distance_bows.png')
plt.close('all')

plt.figure()
sns.heatmap(euclidean_distance_tfidfs)
plt.savefig('euclidean_distance_tfidfs.png')
plt.close('all')
