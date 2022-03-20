import sys
import luima_sbd.sbd_utils
import spacy
import re
import math
import numpy as np
from spacy.symbols import ORTH
import pickle
import nltk
import fasttext
from nltk.corpus import stopwords
import warnings

warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
nlp.tokenizer.add_special_case('Vet. App.', [{ORTH: 'Vet. App.'}])
nlp.tokenizer.add_special_case(u'Pub L. No.', [{ORTH: u'Pub L. No.'}])
nlp.tokenizer.add_special_case(u'Veterans Affairs (VA)', [{ORTH: u'Veterans Affairs (VA)'}])
nlp.tokenizer.add_special_case(u'Regional Office (RO)', [{ORTH: u'Regional Office (RO)'}])
nlp.tokenizer.add_special_case(u'Fed. Reg.', [{ORTH: u'Fed. Reg.'}])
nlp.tokenizer.add_special_case(u'Fed. Cir.', [{ORTH: u'Fed. Cir.'}])


def spacy_tokenize3(txt):
    doc = nlp(txt)
    tokens = list(doc)
    clean_tokens = []
    for t in tokens:
        if t.pos_ == 'PUNCT':
            pass
        elif t.pos_ == 'NUM':
            clean_tokens.append(f'<NUM{len(t)}>')
        else:
            t_ = t.lemma_.encode('ascii', 'ignore').decode('utf-8', 'ignore')
            t_ = t_.lower()
            t_ = re.sub(r'[^a-zA-Z0-9]', '', t_)
            if t_ not in stop_words:
                clean_tokens.append(t_)
    return clean_tokens

def spans_add_spacy_tokens(spans):
    for s in spans:
        s['tokens_spacy'] = spacy_tokenize3(s['txt'])


def average_embedding_vector(spans, model):
    avg_lst = []
    for i in range(len(spans)): #for each sentence
        vec_sum = 0
        for j in range(len(spans[i]['tokens_spacy'])): #for each token
            vec_sum = vec_sum + model.get_word_vector(spans[i]['tokens_spacy'][j])
            #vec_sum = vec_sum + np.sum(model.get_word_vector(spans[i]['tokens_spacy'][j]))
        vec_avg = vec_sum/len(spans[i]['tokens_spacy'])
        avg_lst.append(vec_avg)
    return avg_lst


def normalized_vector(spans, mean, deviation):
    length_list = [len(i['tokens_spacy']) for i in spans]
    for i in range(len(length_list)):
        length_list[i] = (length_list[i] - mean)/deviation

    return length_list


def make_feature_vectors(spans, model, mean, deviation):
    average_vector = np.array(average_embedding_vector(spans, model))
    starts_normalized = np.array([s['start_normalized'] for s in spans])
    normalized_lst = normalized_vector(spans, mean, deviation)
    X = np.concatenate((average_vector,
                        np.expand_dims(starts_normalized, axis=1),
                        np.expand_dims(normalized_lst, axis=1)), axis=1)
    return X


def main():
    args = sys.argv[1:]

    if len(args) == 1:
        path_to_text_file = args[0]
        with open(path_to_text_file, mode='r', encoding='latin-1') as f:
            plainText = f.read()

        # generating a list of annotations from the BVA decision with luima_sbd
        unlabeled = []
        regex = re.compile(r'[\t\r\n]')

        plainText.encode('latin-1', 'ignore')
        plainText = regex.sub(" ", plainText)
        plainText = plainText.replace("\\s+", " ")  # multiple spaces

        annotations = luima_sbd.sbd_utils.text2sentences(plainText)
        annotation_offset = luima_sbd.sbd_utils.text2sentences(plainText, offsets=True)
        annotations_start = [e[0] for e in annotation_offset]
        annotations_end = [e[1] for e in annotation_offset]

        for i in range(len(annotations)):
            annotation = {
                'document': id,
                'start': annotations_start[i],
                'start_normalized': annotations_start[i] / len(plainText),
                'end': annotations_end[i],
                'end_normalized': annotations_end[i] / len(plainText),
                'txt': annotations[i]
            }
            unlabeled.append(annotation)

        spans_add_spacy_tokens(unlabeled)

        # loading the trained fasttext model
        fasttext.FastText.eprint = lambda x: None
        model = fasttext.load_model("fasttext.model")

        # loading the best model
        with open('best_model_embedding.pickle', 'rb') as handle:
            clf = pickle.load(handle)

        # loading the mean and std of the training span
        with open('mean_and_deviation.pickle', 'rb') as handle:
            mean_and_deviation = pickle.load(handle)
            mean = mean_and_deviation[0]
            deviation = mean_and_deviation[1]

        # predicting the annotations
        unlabeled_X = make_feature_vectors(unlabeled, model, mean, deviation)
        unlabeled_X = np.nan_to_num(unlabeled_X)

        pred = clf.predict(unlabeled_X)

        # pretty printing sentence: type
        for i in range(len(unlabeled)):
            sent = unlabeled[i]['txt']
            typ = pred[i]
            print('Sentence ' + str(i) + ':\n' + sent + '\n-> Type: ' + typ)

    else:
        print('Please specify the document path pointing to a BVA decision')


if __name__ == '__main__':
    main()


# python analyze.py unlabeled/0600090.txt
# python analyze.py unlabeled/0600334.txt
# pipreqs ldsi/project --force
# pip freeze > requirements.txt
