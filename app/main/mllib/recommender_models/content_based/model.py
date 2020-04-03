from typing import List, Dict, Tuple, Any

from pymongo import MongoClient

def get_beer_keywords(beer_input, beer_list, corpus_tfidf, text_dict, ntop=20):
    """
    get all the relevant words for beer_input.
    """
    input_beer_keywords = []

    for item in sorted(corpus_tfidf.corpus[beer_list.index(beer_input)], key=lambda x: -x[1])[:ntop]:
        # print item
        # print textDict[item[0]]
        input_beer_keywords.append(text_dict[item[0]])
    return input_beer_keywords


def get_similar_beers(beer_input: str, beer_list: List, ind_sim_mat, ntop=10) -> List:
    """
    get ntop beers similar to beer_input
    """

    # check if beer_input in the database
    try:
        beer_id = beer_list.index(beer_input)
        beer_name_inputted = 1
    except IndexError:
        beer_id = beer_input
        beer_name_inputted = 0
    recommended_beers = []

    # find the beer_input from similarity matrix
    for i, item in enumerate(ind_sim_mat):
        if i == beer_id:
            beer_sim_mat = item

    # sort the beer input similarity matrix and get ntop beers
    for beer in sorted(enumerate(beer_sim_mat), key=lambda x: -x[1])[beer_name_inputted:][:ntop]:
        recommended_beers.append(beer_list[beer[0]])
    return recommended_beers


if __name__ == '__main__':
    # import pickle
    # if 'beer_names' not in locals():
    #     with open('beer_names.json') as f:
    #         beer_names = json.load(f)['beer_name']
    #     print('beer_names loaded')
    # if 'textDict' not in locals():
    #     text_dict2 = pickle.load(open("text_dict.p", "rb"))
    #     print('textDict loaded')
    # if 'textDict' not in locals():
    #     text_dict3 = pickle.load(open("textDict.p", "rb"))
    #     print('textDict loaded')
    # if 'index' not in locals():
    #     ind_sim_mat = pickle.load(open("index.p", "rb"), encoding='latin1')
    #     print('index loaded')
    # if 'corpus_tfidf' not in locals():
    #     corpus_tfidf = pickle.load(open("corpus_tfidf.p", "rb"))
    #     print('corpus_tfidf loaded')

    from app.main.db.util import retrieve_bin_doc

    MONGO_CLIENT = MongoClient('localhost', 27017)
    DB = MONGO_CLIENT['ninkasi']

    beer_names = retrieve_bin_doc(DB, 'beerNames', 'beer_name')
    corpus_tfidf = retrieve_bin_doc(DB, 'corpusTfidf', 'corpus_tfidf')
    text_dict = retrieve_bin_doc(DB, 'textDict', 'text_dict')
    ind_sim_mat = retrieve_bin_doc(DB, 'indexSimMat', 'ind_sim_mat')

    beer = 'Abita Bourbon Street Coffee Stout'
    print(get_beer_keywords(beer, beer_names, corpus_tfidf, text_dict))
    print(get_similar_beers('Abita Bourbon Street Coffee Stout', beer_names, ind_sim_mat))
