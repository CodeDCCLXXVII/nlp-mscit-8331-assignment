import KcbFaqs
from flask import Flask, request, abort
from flask_restplus import Api, Resource, fields
import werkzeug

werkzeug.cached_property = werkzeug.utils.cached_property
app = Flask(__name__)
api = Api(app, version='1.0', title='KCB FAQs classification using ngrams')
ngram_model_utter_input = api.model('RequestUtter', {'requestId': fields.String(required=True,
                                                                                  description='Request Id'),
                                                    'utter': fields.String(required=True, description='User Id')})

ns_ngram_model = api.namespace('kcb_faqs_classification', description="Get classification probability")
ngram_model_output = api.model('Response', {'responseCode': fields.String(required=True, description='Status code'),
                                           'requestId': fields.String(required=True, description='Request Id'),
                                           'results': fields.String(required=True, description='Results')
                                           })


@ns_ngram_model.route('/api/v1/faqs/query', methods=['POST'])
@ns_ngram_model.response(200, 'Response successfully retrieved')
@ns_ngram_model.response(400, 'Invalid params passed')
class Response(Resource):
    @ns_ngram_model.marshal_with(ngram_model_output, code=200)
    @ns_ngram_model.doc('Nice having you query me')
    @ns_ngram_model.expect(ngram_model_utter_input)
    def post(self):
        if not request.json or not 'requestId' in request.json or not 'utter' in request.json:
            abort(404)
        response_code = 500
        prediction_results = KcbFaqs_instance.classify_using_bi_grams_computation(
            request.json['utter'], intent_classes_data)
        if prediction_results is not None:
            response_code = 200
        return {'responseCode': response_code,
                'requestId': request.json['requestId'],
                'results': prediction_results}


if __name__ == '__main__':
    KcbFaqs_instance = KcbFaqs('api')
    kcb_faqs_data = KcbFaqs_instance.load_kcb_faqs_data('data/', 'KCB FAQs.xlsx')
    intent_classes_data = KcbFaqs_instance.extra_intent_text_and_bi_grams(kcb_faqs_data)
    app.run(debug=True, host='0.0.0.0', port='3000')
