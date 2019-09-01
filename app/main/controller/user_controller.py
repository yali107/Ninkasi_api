from typing import Dict, List, Any, Tuple
import json
import pickle

from flask import Flask, jsonify, request
from flask_restplus import Resource, Api, fields
from pymongo import MongoClient

from app.main.model.content_based.model import get_similar_beers, get_beer_keywords
from app.main.db.util import retrieve_bin_doc
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

_user_choice = {
    'beer_name': fields.String(required=True, description='beer name'),
    'preference': fields.Integer(required=True, description='user rating')
}
cfmodel = api.model('Insert_beer_name and ranking', {
    'user_selections': fields.List(fields.Nested(_user_choice))
})

MONGO_CLIENT = MongoClient('localhost', 27017)
DB = MONGO_CLIENT['ninkasi']


@api.route('/api/beerlist')
class BeerList(Resource):
    @api.doc('list of beers')
    def get(self):
        beer_list: List = retrieve_bin_doc(DB, 'beerNames', 'beer_name')
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

        beer_list: List = retrieve_bin_doc(DB, 'beerNames', 'beer_name')
        ind_sim_mat = retrieve_bin_doc(DB, 'indexSimMat', 'ind_sim_mat')

        rec_beers = get_similar_beers(beer_select, beer_list, ind_sim_mat, ntop=10)
        response = jsonify(
            {
                'rec_beers': rec_beers
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
        beer_list: List = retrieve_bin_doc(DB, 'beerNames', 'beer_name')
        corpus_tfidf = retrieve_bin_doc(DB, 'corpusTfidf', 'corpus_tfidf')
        text_dict = retrieve_bin_doc(DB, 'textDict', 'text_dict')

        keywords = get_beer_keywords(beer_select, beer_list, corpus_tfidf, text_dict)
        response = jsonify(
            {
                'beer_keywords': keywords
            }
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

# todo: add collaborative filtering api
# @api.route('/api/cfbeerrec')
# class CollabFilterBeerRec(Resource):
#     @api.expect(_user_choice)
#     def post(self):
#         json_input: json = request.json
#         print(json_input)
#         pass
#


if __name__ == '__main__':
    app.run(debug=True, port=5200)
