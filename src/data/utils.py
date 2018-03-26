from __future__ import print_function

import gensim, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from itertools import combinations

def calculate_coherence( w2v_model, term_rankings ):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations( term_rankings[topic_index], 2 ):
            try:
                pair_scores.append( w2v_model.wv.similarity(pair[0], pair[1]))
            except KeyError:
                pair_scores.append(0)
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)

def get_descriptor( all_terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( all_terms[term_index] )
    return top_terms

def train_w2v_model(abstracts):
    print('Preparing to train word2vec model. Please wait..')
    tokenized_abstracts = list(map(lambda a: gensim.utils.simple_preprocess(a), abstracts))
    w2v_model = gensim.models.Word2Vec(tokenized_abstracts, size=150, min_count=1, sg=1)
    print( "Model has %d terms" % len(w2v_model.wv.vocab) )
    print('Training word2vec complete!\n')
    return w2v_model

def build_tfidf_mat(abstracts, stopwords_extra, nb_features=250):
    print('Preparing to build tfidf matrix. Please wait..')
    custom_stop_words = ENGLISH_STOP_WORDS.union(set(['elsevier', 'rights', 'reserved'])) 

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=nb_features, 
                                        stop_words=custom_stop_words)
    tfidf = tfidf_vectorizer.fit_transform(abstracts)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print("Created %d X %d document-term matrix" % (tfidf.shape[0], tfidf.shape[1]))
    print("Vocabulary has %d distinct terms" % len(tfidf_feature_names))
    print('Building tfidf matrix complete!\n')
    return tfidf, tfidf_feature_names

def topic_model_grid_search(topic_range, tfidf_mat):
    topic_models = []

    print('Preparing to grid search for best topic model. Please wait..')
    # try each value of k
    for k in topic_range: 
        print("Applying NMF for k=%d ..." % k, end='\r', )
        # run NMF
        model = NMF( init="nndsvd", n_components=k ) 
        W = model.fit_transform( tfidf_mat )
        H = model.components_    
        # store for later
        topic_models.append( (k,W,H) )
    print('Grid search for best topic model complete!\n')
    return topic_models

def calc_topic_coherence(topic_models, w2v_model, tfidf_feature_names):
    k_values, coherences = [], []

    print('Preparing to compute coherence for the generated topic models. Please wait..')
    for (k,W,H) in topic_models:
        # Get all of the topic descriptors - the term_rankings, based on top 10 terms
        term_rankings = []
        for topic_index in range(k):
            term_rankings.append( get_descriptor( tfidf_feature_names, H, 
                                                    topic_index, 10 ) )
        # Now calculate the coherence based on our Word2vec model
        k_values.append( k )
        coherences.append( calculate_coherence( w2v_model, term_rankings ) )
        print("K=%02d: Coherence=%.4f" % ( k, coherences[-1] ), end='\r', )

    print('Computing coherence for the generated topic models complete!\n')
    return k_values, coherences
