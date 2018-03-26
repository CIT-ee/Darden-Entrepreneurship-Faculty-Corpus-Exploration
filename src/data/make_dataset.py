from __future__ import print_function

import os, sys, argparse
sys.path.insert(0, os.environ['PROJECT_PATH'])

import pandas as pd
from sklearn.externals import joblib

from config.resources import path_to
from src.data.utils import ( train_w2v_model, build_tfidf_mat, topic_model_grid_search,
                                calc_topic_coherence, get_descriptor )

def build_topic_model(source_dat, path_to_dest, tfidf_max_feats):
    stopword_ls = [ 'elsevier', 'rights', 'reserved' ]
    abstracts = source_dat['AB'].dropna()

    w2v_model = train_w2v_model(abstracts)
    tfidf_mat, tfidf_feat_names = build_tfidf_mat(abstracts, stopword_ls, tfidf_max_feats)

    topic_models = topic_model_grid_search(range(3,50 + 1), tfidf_mat)
    k_values, coherences = calc_topic_coherence(topic_models, w2v_model, tfidf_feat_names)

    best_k_pos = coherences.index(max(coherences))
    best_k = k_values[best_k_pos]

    # get the model that we generated earlier.
    W, H = topic_models[best_k_pos][1:]

    topic_sample_path = path_to_dest.format('tm_sample_{}_topics.txt')
    with open(topic_sample_path.format(best_k), 'wb') as f:
        for topic_index in range(best_k):
            descriptor = get_descriptor( tfidf_feat_names, H, topic_index, 15)
            str_descriptor = ", ".join( descriptor )
            print("Topic %02d: %s" % ( topic_index+1, str_descriptor ), file=f)

    joblib.dump((tfidf_mat, tfidf_feat_names, W, H), path_to_dest.format('model.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='jbv', help='dataset to work on')

    args = parser.parse_args()

    source_dat = pd.read_csv(path_to[args.dataset + '_meta'])
    path_to_dest = path_to['topic_models'].format(args.dataset, '{}')
    path_to_dest_dir = os.path.dirname(path_to_dest)

    if not os.path.exists(path_to_dest_dir):
        print('Directory at {} did not exist. Creating it..'.format(path_to_dest_dir))
        os.makedirs(path_to_dest_dir)

    build_topic_model(source_dat, path_to_dest, 250)
