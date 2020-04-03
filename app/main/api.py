from typing import List, Dict

from flask import Flask, jsonify, request
from flask_restplus import Resource, Api, fields
from pymongo import MongoClient
import numpy as np

from app.main.mllib.recommender_models.content_based.model import get_similar_beers, get_beer_keywords
from app.main.mllib.recommender_models.collaborative_filtering.model import get_beer_rec
from app.main.db.util import retrieve_bin_doc, retrieve_ratings_svd, retrieve_beer_info
app = Flask(__name__)
api = Api(
    app,
    version='1.0.0',
    title='Recommendation API',
    description='Restful API for Collaborate Filtering and Content-Based Recommendation Models'
)
# api = BeerModelDto
cbmodel = api.model('Insert_beer_name', {
    'beer_selected': fields.String(required=True, description='input beer')
})

_user_choice = api.model('Choice', {
    'beer_name': fields.String(required=True, description='beer name'),
    'preference': fields.Integer(required=True, description='user rating')
})
cfmodel = api.model('Insert_beer_name and ranking', {
    'user_selection': fields.List(fields.Nested(_user_choice))
})

MONGO_CLIENT = MongoClient('localhost', 27017)
DB = MONGO_CLIENT['ninkasi']


@api.route('/api/beerlist')
class BeerList(Resource):
    @api.doc('list of beers')
    def get(self):
        beer_list: List = retrieve_bin_doc(DB, 'cbBeerNames', 'beer_name')
        response = jsonify(beer_list)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@api.route('/api/cbbeerrec')
class ContentBasedBeerRec(Resource):
    @api.expect(cbmodel)
    def post(self):
        # params = api.payload
        json_data = request.json
        beer_select = json_data.get('beer_selected')

        beer_list: List = retrieve_bin_doc(DB, 'cbBeerNames', 'beer_name')
        ind_sim_mat = retrieve_bin_doc(DB, 'cbIndexSimMat', 'ind_sim_mat')

        rec_beers = get_similar_beers(beer_select, beer_list, ind_sim_mat, ntop=10)
        rec_beer_infos = retrieve_beer_info(DB, rec_beers)
        response = jsonify(
            {
                'rec_beers': rec_beer_infos
            }
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@api.route('/api/cbbeerkws')
class ContentBasedBeerKeywords(Resource):
    @api.expect(cbmodel)
    def post(self):
        json_input = request.json
        beer_select = json_input.get('beer_selected')
        beer_list: List = retrieve_bin_doc(DB, 'cbBeerNames', 'beer_name')
        corpus_tfidf = retrieve_bin_doc(DB, 'cbCorpusTfidf', 'corpus_tfidf')
        text_dict = retrieve_bin_doc(DB, 'cbTextDict', 'text_dict')

        keywords = get_beer_keywords(beer_select, beer_list, corpus_tfidf, text_dict)
        response = jsonify(
            {
                'beer_keywords': keywords
            }
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@api.route('/api/cfbeerrec')
class CollabFilterBeerRec(Resource):
    @api.expect(cfmodel)
    def post(self):
        json_input = request.json
        user_selection = json_input['user_selection']
        svdpp_mat: np.ndarray = retrieve_ratings_svd(DB)
        beer_dict: Dict = retrieve_bin_doc(DB, 'cfBeerDict', 'beer_dict', type_of='None')

        # test_users2: List[Dict] = [
        #     {'beer_name': 'Yazoo Embrace the Funk Series: Deux Rouges', 'preference': 20},
        #     {'beer_name': 'Iron Hill Bourbon Porter', 'preference': 3},
        # ]

        rec_beers: List[str] = get_beer_rec(user_selection, svdpp_mat, beer_dict)
        rec_beer_infos = retrieve_beer_info(DB, rec_beers)
        response = jsonify(
            {
                'rec_beers': rec_beer_infos
            }
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


if __name__ == '__main__':
    app.run(debug=True, port=5200)
