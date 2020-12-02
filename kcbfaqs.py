import pandas as pd
import nltk as nl
import re
import string
import json
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.feature_extraction.text as sk
import numpy as np


class KcbFaqs:
    def __init__(self, mode, ngrams_count=2):
        self.mode = mode
        self.ngrams_count = ngrams_count
        sns.set(style='whitegrid', palette='muted', font_scale=1.2)
        self.HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
        self.wild_card = ''
        sns.set_palette(sns.color_palette(self.HAPPY_COLORS_PALETTE))
        self.tweets_data_list = []
        self.target_outputs = []
        self.vectorx = None
        print("Initialization ...")

    def load_kcb_faqs_data(self, filepath, filename, file_type='xlsx'):
        if file_type == 'csv':
            df_data = pd.read_csv(filepath + filename)
        else:
            df_data = pd.read_excel(filepath + filename)
        df_data = df_data.loc[df_data['intent'] != 'N/A']
        print(df_data.head())
        print(df_data.tail())
        print(df_data.info())
        print(f'Columns available {df_data.columns}')
        print(f'Unique classifications available {df_data.intent.unique().tolist()}')
        self.visualize(df_data, filepath)
        return df_data

    def visualize(self, df_data, filepath):
        chart = sns.countplot(df_data.intent, palette=self.HAPPY_COLORS_PALETTE)
        plt.title("Number of texts per intent, KCB FAQs")
        chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.savefig(filepath + 'summary.png')
        if str(self.mode).casefold() == 'iicmd':
            plt.show()

    def clean_text(self, text):
        '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers.'''
        text = text.lower()
        text = text.replace('@KCBCare', '').replace('@KCBGroup', '').strip()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    def generate_ngrams(self, text, ngrams_count):
        # text = str(text).split()
        return list(nl.ngrams(text, ngrams_count))

    def generate_pos(self, text):
        pos = []
        for text_pos in nl.pos_tag(text.split()):
            pos.append(text_pos[1])
        return pos

    def get_vector(self):
        return sk.CountVectorizer()

    def intialize_vector(self, data):
        # initialize the vector
        self.vectorx = self.get_vector()
        vectorx_trans = self.vectorx.fit_transform(data)
        # A = vectorx.get_feature_names()
        vectorx_trans_arr = vectorx_trans.toarray()
        return vectorx_trans_arr

    def bipolar(self, input_x):
        return np.where(input_x == 0, -1, input_x)

    def get_weights(self, old_weight, input_x, target):
        input_x = self.bipolar(input_x)
        # print(input_x)
        new_weight = input_x * target
        new_weight = old_weight + new_weight
        return new_weight

    def get_ann_classification(self, final_weight, input_text):
        input_class = 'kcb_m_pesa'
        # vectorx = self.get_vector()
        input_v = self.vectorx.transform(input_text).toarray()
        input_v = self.bipolar(input_v)
        input_class_v = final_weight * input_v
        input_class_sum = np.sum(input_class_v)
        if input_class_sum < 0:
            input_class = 'kcb_account'
        return input_class

    def get_final_weight(self, t_outputs, initial_weight, data_vector):
        init_weight = np.array(initial_weight)
        new_weight = init_weight
        for i in range(len(t_outputs)):
            new_weight = self.get_weights(new_weight, data_vector[i], t_outputs[i])
        return new_weight

    def ann_classification(self, tweet_data, user_input_text):
        vector_x = self.intialize_vector(tweet_data)
        init_weight = []
        for ele in vector_x[0]:
            init_weight.append(0)
        final_weight = self.get_final_weight(self.target_outputs, init_weight, vector_x)
        user_input_text_list = [user_input_text]
        ann_class = self.get_ann_classification(final_weight, user_input_text_list)
        print(f'Ann classfication {ann_class}')
        return ann_class

    def extra_intent_text_and_bi_grams(self, df_data):
        # clean data
        df_data = df_data[df_data.text.notna()]
        df_data.text = df_data['text'].apply(self.clean_text)
        intent_classes = df_data.intent.unique().tolist()
        classes_data = []
        for intent_group in intent_classes:
            intent_group_data = df_data.loc[df_data['intent'] == intent_group]
            intent_group_text = ''
            for index, row in intent_group_data.iterrows():
                intent_group_text = intent_group_text + ' ' + row.text + self.wild_card
                self.tweets_data_list.append(row.text)
                self.target_outputs.append(row.label)
            intent_group_text_pos_list = self.generate_pos(intent_group_text)
            intent_group_text_pos = ' '.join(intent_group_text_pos_list)
            intent_group_dict = {'intent_group': intent_group,
                                 'intent_group_text': intent_group_text,
                                 'intent_group_text_pos': intent_group_text_pos,
                                 'intent_group_n_grams': self.generate_ngrams(intent_group_text.split(),
                                                                              self.ngrams_count),
                                 'intent_group_pos_grams': self.generate_ngrams(intent_group_text_pos_list,
                                                                                self.ngrams_count)}
            classes_data.append(intent_group_dict)
        # print(classes_data)
        return classes_data

    def classify_using_bi_grams_computation(self, text, classes_data):
        text = self.wild_card + text + self.wild_card
        text = self.clean_text(text)
        text_list = text.split()
        bi_input = list(nl.ngrams(text_list, self.ngrams_count))
        pos_input = self.generate_pos(text)
        pos_input_ngrams = self.generate_ngrams(pos_input, self.ngrams_count)
        intent_classification = []
        for intent_group_dict_item in classes_data:
            for index, input_tag in enumerate(bi_input):
                ngram_prob = 1
                pos_prob = 1
                try:
                    ngram_prob *= intent_group_dict_item['intent_group_n_grams'].count(input_tag) / \
                        intent_group_dict_item['intent_group_text'].count(input_tag[0])
                except:
                    ngram_prob = 0.00

                try:
                    pos_prob *= intent_group_dict_item['intent_group_pos_grams'].count(pos_input_ngrams[index]) / \
                        intent_group_dict_item['intent_group_text_pos'].count(pos_input_ngrams[index][0])
                except:
                    pos_prob = 0.00

            ann_classification_class = self.ann_classification(self.tweets_data_list, text)
            if ann_classification_class != intent_group_dict_item['intent_group']:
                ann_classification_class = 'N/A'

            intent_classification_item = {"intent": intent_group_dict_item['intent_group'],
                                          "probability_n_gram": ngram_prob,
                                          "probability_pos": pos_prob,
                                          "probability_ann": ann_classification_class
                                          }
            intent_classification.append(intent_classification_item)
        return intent_classification

    def print_pretty(self, response):
        parsed = json.dumps(response, indent=4, sort_keys=True)
        if str(self.mode).casefold() == 'cmd':
            print(parsed)
        return parsed


