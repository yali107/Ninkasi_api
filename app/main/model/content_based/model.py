from typing import List, Dict, Tuple, Any


def get_beer_keywords(beer_input, corpus_tfidf, beer_list, textDict, ntop=20):
    """
    get all the relevant words for beer_input.
    """
    input_beer_keywords = []

    for item in sorted(corpus_tfidf.corpus[beer_list.index(beer_input)], key=lambda x: -x[1])[:ntop]:
        # print item
        # print textDict[item[0]]
        input_beer_keywords.append(textDict[item[0]])
    return input_beer_keywords


def get_similar_beers(beer_input, beer_list, index, ntop=10) -> List:
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
    for i, item in enumerate(index):
        if i == beer_id:
            beer_sim_mat = item

    # sort the beer input similarity matrix and get ntop beers
    for beer in sorted(enumerate(beer_sim_mat), key=lambda x: -x[1])[beer_name_inputted:][:ntop]:
        recommended_beers.append(beer_list[beer[0]])
    return recommended_beers


if __name__ == '__main__':
    import pickle
    if 'beer_list' not in locals():
        beer_list = pickle.load(open("beer_list.p", "rb"))
        print('beer_list loaded')
    if 'textDict' not in locals():
        textDict = pickle.load(open("text_dict.p", "rb"))
        print('textDict loaded')
    if 'index' not in locals():
        index = pickle.load(open("index.p", "rb"), encoding='latin1')
        print('index loaded')
    if 'corpus_tfidf' not in locals():
        corpus_tfidf = pickle.load(open("corpus_tfidf.p", "rb"))
        print('corpus_tfidf loaded')

    beer = 'Abita Bourbon Street Coffee Stout'
    print(get_beer_keywords(beer, corpus_tfidf, beer_list, textDict))