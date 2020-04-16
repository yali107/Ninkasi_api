from typing import List, Dict, Tuple, Any

from pymongo import MongoClient
from bson.binary import Binary
import pickle
import json
import numpy as np

CLIENT = MongoClient('localhost', 27017)
DB = CLIENT['ninkasi']
NUM_COLS = 1269


def insert_collection(pickle_name, query_name) -> None:

    coll = DB.textDict
    with open('../model/content_based/' + pickle_name + '.p', 'rb') as f:
        pickled_data = pickle.load(f, encoding='latin1')

    thebytes = pickle.dumps(pickled_data)
    coll.insert_one({query_name: Binary(thebytes)})


def retrieve_bin_doc(db, coll_name: str, doc_name: str, type_of='pickle'):
    coll = db[coll_name]
    cursor = coll.find({})
    for doc in cursor:
        if doc.get(doc_name):
            if isinstance(doc.get(doc_name), bytes):
                if type_of == 'pickle':
                    return _transform_doc(doc.get(doc_name), type_of='pickle')
                elif type_of == 'npy':
                    return _transform_doc(doc.get(doc_name), type_of='npy')
                elif type_of == 'str':
                    return _transform_doc(doc.get(doc_name), type_of='str')
                else:
                    raise TypeError('Only accept pickle and npy data')
            else:
                return _transform_doc(doc.get(doc_name), type_of='dict')


def retrieve_ratings_svd(db) -> np.ndarray:
    doc_suffix: str = 'ABCDEFGHIJKLM'
    ratings_arr = None
    for s in doc_suffix:
        temp_arr: np.ndarray = retrieve_bin_doc(db, 'cfRatingsSvdpp_'+s, 'ratings_svdpp', type_of='npy')
        temp_arr = np.reshape(temp_arr, (-1, NUM_COLS))
        if ratings_arr is not None:
            ratings_arr = np.concatenate((ratings_arr, temp_arr))
        else:
            ratings_arr = temp_arr
    return ratings_arr


def retrieve_beer_info(db, beer_names):
    if len(beer_names) == 0:
        return None
    beer_info = retrieve_bin_doc(db, 'beerInfo', 'beer_info', type_of='str')
    return [{
        'beer_name': i,
        'state': beer_info[i][0],
        'avg_user_rating': beer_info[i][1],
        'avg_overall_rating': beer_info[i][2],
        'detail_link': beer_info[i][3]
    } for i in beer_names]


def _transform_doc(doc_content, type_of='pickle'):
    """
    transform doc (binary, json) in mongodb to proper object
    :param doc_content:
    :param bin:
    :return:
    """
    if type_of == 'pickle':
        return pickle.loads(doc_content)
    elif type_of == 'npy':
        return np.frombuffer(doc_content)
    elif type_of == 'dict':
        return doc_content
    elif type_of == 'str':
        return json.loads(doc_content)
    else:
        raise TypeError




if __name__ == '__main__':
    # insert_collection('corpus_tfidf', 'corpus_tfidf')
    # insert_collection('index', 'ind_sim_mat')

    # with open('../model/content_based/text_dict.p', 'rb') as f:
    #     text_dict = dict(pickle.load(f))
    #
    # print(text_dict)
    # with open('../model/content_based/text_dict.json', 'w') as f:
    #     json.dump(text_dict, f)
    # coll = db.indexSimMat
    # cursor = coll.find({})
    # for doc in cursor:
    #     if doc.get('ind_sim_mat'):
    #         data_bin = doc['ind_sim_mat']
    #         data = pickle.loads(data_bin)
    #     print(doc)
    # ratt_arr = retrieve_ratings_svd(DB)
    # file_arr = np.load('C:\\LiYuan\\personal-projects\\Ninkasi\\api\\app\\main\\mllib\\data\\cf\\ratings_svdpp.npy')
    # print(ratt_arr)
    # corpus_tfidf = retrieve_bin_doc(DB, 'cbCorpusTfidf', 'corpus_tfidf')
    ind_sim_mat = retrieve_bin_doc(DB, 'cbIndexSimMat', 'ind_sim_mat')
    text_dict = retrieve_bin_doc(DB, 'cbTextDict', 'text_dict')
    beer_dict: Dict = retrieve_bin_doc(DB, 'cfBeerDict', 'beer_dict', type_of='None')
    with open('../mllib/data/cb/ind_sim_mat.p', 'wb') as f:
        pickle.dump(ind_sim_mat, f)
    print(ind_sim_mat)

    # print(retrieve_beer_info(DB, ['(512) Cascabel Cream Stout', '1809 Berliner Style Weisse Zymatore - Gin & Pinot Noir Barrels']))
