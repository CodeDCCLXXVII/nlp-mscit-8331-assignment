import KcbFaqs

if __name__ == '__main__':
    KcbFaqs_instance = KcbFaqs('CMD')
    kcb_faqs_data = KcbFaqs_instance.load_kcb_faqs_data('data/',
                                                        '20201118_123358_kcbcare_tweets_classified_cleaned.csv', 'csv')
    intent_classes_data = KcbFaqs_instance.extra_intent_text_and_bi_grams(kcb_faqs_data)
    input_text = input("Enter your KCB question: ")
    KcbFaqs_instance.print_pretty(KcbFaqs_instance.classify_using_bi_grams_computation(input_text, intent_classes_data))