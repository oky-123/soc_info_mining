import nltk
import collections
from nltk.corpus import wordnet as wn #lemmatize関数のためのimport

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("reuters")
nltk.download("punkt")

docs=[
# Emerald Sword/Rhapsody of Fire
"acrossed the valleys the dust of midlands to search for the third key to open the gates.",
"Now I'm near the altar the secret inside as legend told my beloved sun light the dragons' eyes.",
"On the way to the glory I'll honour my sword to serve right ideals and justice for all.",
"Finally happened the sun lit their eyes the spell was creating strange games of light.",
"Thanks to hidden mirrors I found my lost way over the stones I reached the place it was a secret cave.",
"In a long bloody battle that prophecies told the light will prevail hence wisdom is gold.",
"For the king for the land for the mountains for the green valleys where dragons fly for the glory the power to win the black lord I will search for the emerald sword.",
"Only a warrior with a clear heart could have the honour to be kissed by the sun.",
"Yes, I'm that warrior I followed my way led by the force of cosmic soul I can reach the sword.",
"On the way to the glory I'll honour my sword to serve right ideals and justice for all.",
"For the king for the land for the mountains.",
"For the green valleys where dragons fly.",
"For the glory the power to win the black lord.",
"I will search for the emerald sword.",
"Ten, nine, eight, seven, six, five, four, three, two, one.",

# The Final Countdown/Europe
"We're leaving together, but still it's farewell.",
"And maybe we'll come back to earth, who can tell?",
"I guess there is no one to blame.",
"We're leaving ground leaving ground.",
"Will things ever be the same again?",
"It's the final countdown.",
"The final countdown.",
"Oh.",
"We're heading for Venus and still we stand tall.",
"'Cause maybe they've seen us.",
"And welcome us all, yeah.",
"With so many light years to go and things to be found.",
"I'm sure that we'll all miss her so.",

# through the fire and flames/DragonForce
"On a cold winter morning in the time before the light.",
"In flames of Death's eternal reign we ride towards the fight.",
"When the darkness has fallen down and the times are tough, alright.",
"The sound of evil laughter falls around the world tonight.",
"Fighting hard, fighting on for the steel through the wastelands evermore.",
"The scattered souls will feel the hell.",
"Bodies wasted on the shores.",
"On the blackest plains in hell's domain.",
"We watch them as they go in fire and pain, now once again.",
"So now, we fly ever free, we're free before the thunderstorm on towards the wilderness, our quest carries on.",
"Far beyond the sundown, far beyond the moonlight deep inside our hearts and all our souls.",
"So far away, we wait for the day for the lives, all so wasted and gone.",
"We feel the pain of a lifetime lost in a thousand days through the fire and the flames, we carry on.",
"As the red day is dawning and the lightning cracks the sky.",
"They'll raise their hands to the heavens above with resentment in their eyes.",
"Running back through the mid-morning light, there's a burning in my heart.",
"We're banished from a time in a fallen land to a life beyond the stars.",
"In your darkest dreams, see to believe our destiny this time.",
"And endlessly we'll all be free tonight.",
"And on the wings of a dream, so far beyond reality.",
"All alone in desperation, now the time has gone.",
"Lost inside you'll never find, lost within my own mind.",

# Sunday Morning/Maroon5
"Sunday morning, rain is falling",
"Steal some covers, share some skin",
"Clouds are shrouding us in moments unforgettable",
"You twist to fit the mold that I am in",
"But things just get so crazy",
"Living life gets hard to do",
"And I would gladly hit the road, get up and go if I knew",
"That someday it would lead me back to you",
"That may be all I need",
"In darkness, she is all I see",
"Come and rest your bones with me",
"Driving slow on Sunday morning",
"And I never want to leave",
"Fingers trace your every outline",
"Paint a picture with my hands",
"Back and forth we sway like branches in a storm",
"Change the weather still together when it ends",
"That may be all I need",
"In darkness she is all I see",
"Come and rest your bones with me",
"Driving slow on Sunday morning",
"And I never want to leave",
"But things just get so crazy, living life gets hard to do",
"Sunday morning, rain is falling and I'm calling out to you",
"Singing someday it'll bring me back to you",
"Find a way to bring myself back home to you",
"And you may not know",
"That may be all I need",
"In darkness she is all I see",
"Oh, come and rest your bones with me",
"Driving slow, driving slow (all I need, all I see)",
"oh yeah yeah oh yeah yeah oh yeah yeah (bones with me)",
"I'm a flower in your hair",
"yeah yeah, yeah yeah",

# Sugar/Maroon5
"I'm hurting baby, I'm broken down",
"I need your loving, loving I need it now",
"When I'm without you, I'm something weak",
"You got me begging, begging I'm on my knees",
"I don't wanna be needing your love",
"I just wanna be deep in your love",
"And it's killing me when you're away",
"Ooh baby",
"'Cause I really don't care where you are",
"I just wanna be there where you are",
"And I gotta get one little taste",
"Babe, my broken pieces, you pick them up",
"Don't leave me hanging, hanging",
"Come give me some",
"When I'm without you, I'm so insecure",
"You are the one thing, one thing I'm living for",
"Yeah",
"I want that red velvet",
"I want that sugar sweet",
"Don't let nobody touch it unless that somebody's me",
"I gotta be a man, there ain't no other way",
"'Cause girl you're hotter than a Southern California day",
"Never wanna play no games, you don't gotta be afraid",
"Don't give me all that shy shit",
"No makeup on, that's my sugar",
]

# ストップワード
en_stop = nltk.corpus.stopwords.words('english')

# 前処理用メソッド
def preprocess_word(word, stopwordset):
    #1.make words lower ex: Python =>python
    word = word.lower()
    #2.remove "," and "."
    if word in [",", "."] or not word:
        return None
    #3.remove stopword  ex: the => (None)
    if word in stopwordset:
        return None
    #4.lemmatize  ex: cooked=>cook
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    elif lemma in stopwordset: #lemmatizeしたものがstopwordである可能性がある
        return None
    else:
        return lemma

import re
def preprocess_document(document):
    document = re.split("[ ,.]", document)
    document = [preprocess_word(w, en_stop) for w in document]
    document = [w for w in document if w is not None]
    return document

def preprocess_documents(documents):
    return [preprocess_document(document) for document in documents]

# 前処理
pre_docs = preprocess_documents(docs)
pre_docs = [" ".join(doc) for doc in pre_docs]

# LDAの準備
import gensim
from gensim import corpora

# 辞書の作成
dictionary = corpora.Dictionary(preprocess_documents(docs))

# コーパスの作成
import sys
corpus_ = [dictionary.doc2bow(doc) for doc in preprocess_documents(docs)]

ldamodel = gensim.models.ldamodel.LdaModel(
    corpus=corpus_,
    num_topics=int(sys.argv[1]),
    id2word=dictionary,
    alpha=0.1,                             #optional LDAのハイパーパラメータalpha
    eta=0.1,                                 #optional LDAのハイパーパラメータbeta
    #minimum_probability=0.0    #optional 学習結果に格納するトピック・単語の生起確率の下限
)

# 可視化
import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus_, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
pyLDAvis.save_html(lda_display,'vis_'+sys.argv[1]+'.html')
