import pandas as pd
import nltk as nl
import re
import string
import json
import seaborn as sns
import matplotlib.pyplot as plt


class KcbFaqs:
    def __init__(self, mode, ngrams_count=2):
        self.mode = mode
        self.ngrams_count = ngrams_count
        sns.set(style='whitegrid', palette='muted', font_scale=1.2)
        self.HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
        self.wild_card = ''
        sns.set_palette(sns.color_palette(self.HAPPY_COLORS_PALETTE))
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
        if str(self.mode).casefold() == 'cmd':
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
        text = str(text).split()
        return list(nl.ngrams(text, ngrams_count))

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
            intent_group_dict = {'intent_group': intent_group,
                                 'intent_group_text': intent_group_text,
                                 'intent_group_bi_grams': self.generate_ngrams(intent_group_text, self.ngrams_count)}
            classes_data.append(intent_group_dict)
        # print(classes_data)
        return classes_data

    def classify_using_bi_grams_computation(self, text, classes_data):
        text = self.wild_card + text + self.wild_card
        bi_input = list(nl.ngrams(self.clean_text(text).split(), self.ngrams_count))
        intent_classification = []
        for intent_group_dict_item in classes_data:
            for input_tag in bi_input:
                x = 1
                try:
                    x *= intent_group_dict_item['intent_group_bi_grams'].count(input_tag) / \
                        intent_group_dict_item['intent_group_text'].count(input_tag[0])
                except:
                    x = 0.00
            intent_classification_item = {"intent": intent_group_dict_item['intent_group'], "probability": x}
            intent_classification.append(intent_classification_item)
        return intent_classification

    def print_pretty(self, response):
        parsed = json.dumps(response, indent=4, sort_keys=True)
        if str(self.mode).casefold() == 'cmd':
            print(parsed)
        return parsed


