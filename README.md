# LDSI: Legal Data Science &amp; Informatics

In this project, a search engine for the US Board of Veterans’ Appeals (BVA) is introduced.
The goal is to propose a system that can extract reasoning for case outcomes. The designed
model uses a law specific sentence segmenter as preprocessing and tested with several
different machine learning models.

The sentence types in this project are:
- CaseHeader
- CaseIssue
- Citation
- ConclusionOfLaw
- Evidence
- EvidenceBasedOrIntermediateFinding
- EvidenceBasedReasoning
- Header
- LegalRule
- LegislationAndPolicy
- PolicyBasedReasoning
- Procedure
- RemandInstructions

## Understanding analyze.py
The *analyze.py* with Python version 3.9.2 is created as well as *requirements.txt*. 

An example output for the command line follows `python analyze.py [directory]/0600334.txt` after running `pip install -r requirements.txt`

## Source Code Explanation
The source code that is in the LDSI_project.ipynb corresponds to the data prepatation, data analysis, sentence segmentation, predictions, word embedding and TF-IDF based model, basically every aspect of a data anayltics project is covered. 

### Phase 0: Setup
Basic setup, importing necessary libraries, setting the directory, the corpus path and loading the data.

### Phase 1: Dataset Splitting
The annotated 141 BVA decisions (70 granted, 71 denied) are split to create a balanced dataset: 
- %10 test set which corresponds to 7 granted and 7 denied decisions
- %10 dev set which corresponds to 7 granted and 7 denied decisions
- %80 train set which corresponds to 56 granted and 57 denied decisions

### Phase 2: Deciding on a Sentence Segmenter
The anotated cases are here used to decide which sentence segmenter to use. 
- Standard segmentation using Spacy
- Improved sentence segmentation using Spacy with special cases like 'Vet. App', 'Fed. Cir.', 'Pub. L. No.'
- Savelka's law-specificsentence segmenter *(https://github.com/jsavelka/luima_sbd)*

Comparing these 3 methods, Savelka's law-specific sentence segmenter is the best choice to use. 

### Phase 3: Preprocessing
- Sentence segmentation of the unlabeled dataset with Savelka's law-specific sentence segmenter. 
- Sentence-wise preprocessing to treat punctuation, simplify numbers, lowercasing, removing all non ASCII characters, etc. 
- Use the sentence-wise preprocessing to tokenize the unlabeled data.

### Phase 4: Developing Word Embeddings
Using FastText, the unlabeled data is trained to create word vector representation. It is trained for 25 epochs, with minimum word occurrence count 20 and minimum length of char n-gram 2, creating a vocabulary size of around 14k words. 
Some example words are chosen to be explored using nearest neighbors: veteran, ptsd, korea, granted, denied, tinnitus, etc.

### Phase 5: Training Classifiers
There are 2 models that are first created: TF-IDF based and FastText word embedding based model 
Both models are trained with Linear and Non Linear models like Linear SVM, Decision Trees, Random Forests and Multi-layer Perceptron.

The best performing classifier is the Multi-layer Perceptron classifier for both the TF-IDF and the embedding based model, whereas the embedding based model achieved better results compared to TF-IDF with 0.86 accuracy on the development set, trained with ADAM as optimizer and with early stopping to prevent overfitting.

### Phase 6: Error Analysis
The worst performing classification types are analyzed to better understand the reason. 
In the test set, Legislation and Policy are the most misclassified type. This is due to the ambiguous form of legislation and policies and also due to the fact that there are not enough annotations of this type. The same reasoning applies to Policy based Reasoning.

### Phase 7: Discussion & Lessons Learned
Overall, the project was about performing an experiment for understanding BVA cases for examining the underlying reasons for case outcomes as well as sentence level classification for annotation purposes.

### Phase 8: Code Deliverable
It includes the best model as a program callable from the command line and take a single argument pointing to a text file containing a BVA decision.


