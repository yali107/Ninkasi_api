from pymongo import MongoClient

from bson.binary import Binary
import pickle
import json

client = MongoClient('localhost', 27017)
db = client['ninkasi']


def insert_collection(pickle_name, query_name):

    coll = db.textDict
    with open('../model/content_based/' + pickle_name + '.p', 'rb') as f:
        pickled_data = pickle.load(f, encoding='latin1')

    thebytes = pickle.dumps(pickled_data)
    coll.insert_one({query_name: Binary(thebytes)})


def retrieve_bin_doc(db, coll_name, doc_name):
    coll = db[coll_name]
    cursor = coll.find({})
    for doc in cursor:
        if doc.get(doc_name):
            if isinstance(doc.get(doc_name), bytes):
                return _transform_doc(doc.get(doc_name), bin=True)
            else:
                return _transform_doc(doc.get(doc_name))


def _transform_doc(doc_content, bin=False):
    """
    transform doc (binary, json) in mongodb to proper object
    :param doc_content:
    :param bin:
    :return:
    """
    if bin:
        return pickle.loads(doc_content)
    else:
        return doc_content


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
    insert_collection('text_dict', 'text_dict')
    # print(retrieve_bin_doc(db, 'textDict', 'text_dict'))

