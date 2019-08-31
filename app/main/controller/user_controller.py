from typing import Dict, List, Any, Tuple
import json
import pickle

from flask import Flask, jsonify, request
from flask_restplus import Resource, Api, fields

from app.main.model.content_based.model import get_similar_beers, get_beer_keywords
# from app.main.dto import BeerModelDto

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

@api.route('/api/beerlist')
class BeerList(Resource):
    @api.doc('list of beers')
    def get(self):
        with open('../model/content_based/beer.json') as f:
            beer_list: Dict = json.load(f)
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
        with open('../model/content_based/beer.json') as f:
            beer_list: Dict = json.load(f).get('beer_name')

        with open('../model/content_based/index.p', 'rb') as f:
            index = pickle.load(f, encoding='latin1')

        # keywords = get_beer_keywords(beer_select, corpus_tfidf, beer_list, text_dict)
        rec_beers = get_similar_beers(beer_select, beer_list, index, ntop=10)
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
        with open('../model/content_based/beer.json') as f:
            beer_list: Dict = json.load(f).get('beer_name')

        with open('../model/content_based/corpus_tfidf.p', 'rb') as f:
            corpus_tfidf = pickle.load(f)

        with open('../model/content_based/text_dict.p', 'rb') as f:
            text_dict = pickle.load(f)

        keywords = get_beer_keywords(beer_select, corpus_tfidf, beer_list, text_dict)
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
