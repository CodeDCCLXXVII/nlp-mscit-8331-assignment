import KcbFaqs
import pandas as pd


def classify_tweet(tweet):
    input_text = tweet.replace('@KCBCare', '').replace('@KCBGroup', '').replace("\n", '').strip()
    input_text = KcbFaqs_instance.clean_text(input_text)
    tweet_class = 'N/A'
    if len(input_text.split()) > 1:
        print(input_text)
        class_response = KcbFaqs_instance.classify_using_bi_grams_computation(input_text, intent_classes_data)
        print(class_response)
        for class_response_item in class_response:
            print(class_response_item)
            probability = class_response_item["probability"]
            print(probability)
            if probability > 0.25:
                tweet_class = class_response_item["intent"]
    print(tweet_class)
    return tweet_class


if __name__ == '__main__':
    KcbFaqs_instance = KcbFaqs('Classify')
    kcb_faqs_data = KcbFaqs_instance.load_kcb_faqs_data('data/', '20201118_123358_kcbcare_tweets_classified.csv', 'csv')
    intent_classes_data = KcbFaqs_instance.extra_intent_text_and_bi_grams(kcb_faqs_data)
    df_tweets = pd.read_csv('data/20201118_142331_kcbcare_tweets.csv')
    df_tweets['intent'] = df_tweets['text'].apply(classify_tweet)
    df_tweets.to_csv('data/20201118_142331_kcbcare_tweets_classified.csv')
