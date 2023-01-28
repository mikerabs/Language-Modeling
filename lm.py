from math import log, exp
from collections import defaultdict
import argparse

from numpy import mean

import nltk
from nltk import FreqDist
from nltk.util import bigrams
from nltk.tokenize import TreebankWordTokenizer

kLM_ORDER = 2
kUNK_CUTOFF = 3#3
kNEG_INF = -1e6

kSTART = "<s>"
kEND = "</s>"

def lg(x):
    return log(x) / log(2.0)

class BigramLanguageModel:
    def __init__(self, unk_cutoff, jm_lambda=0.6, add_k=0.1,
                 katz_cutoff=5, kn_discount=0.1, kn_concentration=1.0,
                 tokenize_function=TreebankWordTokenizer().tokenize,
                 normalize_function=lambda x: x.lower()):
        self._unk_cutoff = unk_cutoff
        self._jm_lambda = jm_lambda
        self._add_k = add_k
        self._katz_cutoff = katz_cutoff
        self._kn_concentration = kn_concentration
        self._kn_discount = kn_discount
        self._vocab_final = False

        self._tokenizer = tokenize_function
        self._normalizer = normalize_function
        
        # Add your code here!
        self._countWords = {}#our dictionary for train_seen()
        self._countWords2 = {}#dictionary for add_train()

    def train_seen(self, word, count=1):#count used to be = 1
        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        # Add your code here!  
        # Modify this function so that it will keep track of all of 
        # the tokens in the training corpus and their counts. 

        #use dictionary
        countEach = 0
        if word in self._countWords:
            countEach += self._countWords[word]
        elif count !=1:
            countEach += count
            countEach -= 1# because we always count it

        countEach += 1    
        self._countWords[word] = countEach


    def tokenize(self, sent):
        """
        Returns a generator over tokens in the sentence.  

        You don't need to modify this code.
        """
        for ii in self._tokenizer(sent):
            yield ii
        
    def vocab_lookup(self, word):
        """
        Given a word, provides a vocabulary representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"

        # Add your code here

        #check if the searched for word is in the dictionary first

        if word in self._countWords and self._countWords[word] >= self._unk_cutoff:
            return word
        elif word == kSTART or word == kEND:
            return word
        return "unknown"# the common unknown identifier that will be returned if word count does not meet threshold

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """

        # You probably do not need to modify this code
        self._vocab_final = True

    def tokenize_and_censor(self, sentence):
        """
        Given a sentence, yields a sentence suitable for training or
        testing.  Prefix the sentence with <s>, replace words not in
        the vocabulary with <UNK>, and end the sentence with </s>.

        You should not modify this code.
        """
        yield self.vocab_lookup(kSTART)
        for ii in self._tokenizer(sentence):
            yield self.vocab_lookup(self._normalizer(ii))
        yield self.vocab_lookup(kEND)


    def normalize(self, word):
        """
        Normalize a word

        You should not modify this code.
        """
        return self._normalizer(word)


    def mle(self, context, word):
        """
        Return the log MLE estimate of a word given a context.  If the
        MLE would be negative infinity, use kNEG_INF
        """

        #p(hope|I) = count(I hope)/count(hope) -> change add_train() to count the bigrams

        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.

        bigramPair = context + " " + word
        
        if bigramPair not in self._countWords2:
            return kNEG_INF
        
        if self._countWords2[context] == 0:
            return kNEG_INF
        
        

        logmle = lg(self._countWords2[bigramPair]/self._countWords2[context])

        return logmle

    def laplace(self, context, word):
        """
        Return the log MLE estimate of a word given a context.
        """

        bigramPair = context + " " + word
        
        if bigramPair not in self._countWords2 and context not in self._countWords2:
            return lg(1)/lg(len(self._countWords))
        if bigramPair not in self._countWords2 and context in self._countWords2:
            return lg(1/(self._countWords2[context]+len(self._countWords)))
            
        loglapl = lg((self._countWords2[bigramPair]+1)/(self._countWords2[context]+len(self._countWords)))#+k*vocab_size?
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        return loglapl

    def jelinek_mercer(self, context, word):
        """
        Return the Jelinek-Mercer log probability estimate of a word
        given a context; interpolates context probability with the
        overall corpus probability.
        """
        bigramPair = context + " " + word

        if bigramPair not in self._countWords2 and context not in self._countWords2:
            return 0.0

        
        
        
        
        
        #counting the total wordTokens not including the start token
        countWordTokensNoStart = 0
        for i in self._countWords2:
            isBigram = False
            if i != kSTART:
                for j in i:
                    if j == " ":
                        isBigram = True
                if isBigram == True:
                    continue
                countWordTokensNoStart += self._countWords2[i]

        #problem with counting word tokens in countWords2, the bigrams are also in there, so skip over them
        unigramprobnum = self._countWords2[word]
        unigramprob = unigramprobnum/countWordTokensNoStart
        
        #second case where context exists in countWords2 but the bigram pair is not in countWords2
        if bigramPair not in self._countWords2 and context in self._countWords2:
            return lg((1-self._jm_lambda) * (unigramprob))

        #mle used in this calulation, below is unlogged value, check for bigram pair and context in countWords2 before this      
        mle = self._countWords2[bigramPair]/self._countWords2[context]

        return lg(self._jm_lambda*(mle) + (1-self._jm_lambda) * (unigramprob))
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        return 0.0

    def kneser_ney(self, context, word):
        """
        Return the log probability of a word given a context given
        Kneser Ney backoff
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        return 0.0

    def add_k(self, context, word):
        """
        Additive smoothing, i.e. add_k smoothing, assuming a fixed k
        hyperparameter.
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.

        bigramPair = context + " " + word
    
        if bigramPair not in self._countWords2 and context not in self._countWords2:
                return lg(self._add_k/self._add_k*len(self._countWords))
        if bigramPair not in self._countWords2 and context in self._countWords2:
            return lg(self._add_k/(self._countWords2[context]+self._add_k*len(self._countWords)))

        logaddk = lg((self._countWords2[bigramPair]+self._add_k)/(self._countWords2[context]+self._add_k*len(self._countWords)))

        return logaddk

    def add_train(self, sentence):
        """
        Add the counts associated with a sentence.
        """
        # Directions: Modify so that given a sentence it keeps track 
        # of the necessary counts you’ll need for the probability functions later

        # You'll need to complete this function, but here's a line of
        # code that will hopefully get you started.
        #for each bigram word pairing, (context, word), 
        for context, word in bigrams(self.tokenize_and_censor(sentence)):
            
            countBi = 0
            # account for case where word that isn't part of vocabulary 
            #if(context == "unknown" or  word == "unknown"):
            #    continue

            bigramPair = context + " " + word # the bigram pair we want to count
            if bigramPair in self._countWords2:
                countBi += self._countWords2[bigramPair]#if it is already in the countWords dict, set countBi = to current count
            
            countBi += 1#add one for this observation
            self._countWords2[bigramPair] = countBi#reset countWords2/create new countWords2 instance

###########################################################################################################
            
            countEach = 0
            #if the <s> is part of vocab and already in the countWords2 dictionary, either second instance case/another instance in next sentence
            if context == kSTART:
                if kSTART not in  self._countWords:# add <s> to vocab
                    self._countWords[kSTART] = 1
                if kSTART in self._countWords2:
                    countEach += self._countWords2[context] #set countEach= current number of times word has already occurred
                #regardless if the context is already in countWords2 it is getting a count
                countEach +=1    
                self._countWords2[kSTART]= countEach

            

            if self.vocab_lookup(word) not in self._countWords:#the problem with laplace
                self._countWords[self.vocab_lookup(word)] = 1
            
            
            #reset countEach
            countEach = 0


            #if the word is part of vocab and already in the countWords2 dictionary, either second instance case/another instance in next sentence
            if word in self._countWords and word in self._countWords2:
                countEach += self._countWords2[word] #set countEach= current number of times word has already occurred

            #regardless if the word is already in countWords2 it is getting a count
            countEach +=1    
            self._countWords2[word]= countEach
            
            #if count > 1:#PROBLEM,NO WORDS GET OVER 1
                #print(word,count)
            #print(context, word)

            

    def perplexity(self, sentence, method):
        """
        Compute the perplexity of a sentence given a estimation method

        You do not need to modify this code.
        """
        return 2.0 ** (-1.0 * mean([method(context, word) for context, word in \
                                    bigrams(self.tokenize_and_censor(sentence))]))

    def sample(self, method, samples=25):
        """
        Sample words from the language model.
        
        @arg samples The number of samples to return.
        """
        # Modify this code to get extra credit.  This should be
        # written as an iterator.  I.e. yield @samples times followed
        # by a final return, as in the sample code.

        for ii in xrange(samples):
            yield ""
        return

# You do not need to modify the below code, but you may want to during
# your "exploration" of high / low probability sentences.
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--jm_lambda", help="Parameter that controls " + \
                           "interpolation between unigram and bigram",
                           type=float, default=0.6, required=False)
    argparser.add_argument("--add_k", help="Add k value " + \
                           "for pseudocounts",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--unk_cutoff", help="How many times must a word " + \
                           "be seen before it enters the vocabulary",
                           type=int, default=2, required=False)    
    argparser.add_argument("--katz_cutoff", help="Cutoff when to use Katz " + \
                           "backoff",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--lm_type", help="Which smoothing technique to use",
                           type=str, default='mle', required=False)
    argparser.add_argument("--brown_limit", help="How many sentences to add " + \
                           "from Brown",
                           type=int, default=-1, required=False)
    argparser.add_argument("--kn_discount", help="Kneser-Ney discount parameter",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--kn_concentration", help="Kneser-Ney concentration parameter",
                           type=float, default=1.0, required=False)
    argparser.add_argument("--method", help="Which LM method we use",
                           type=str, default='laplace', required=False)
    
    args = argparser.parse_args()    
    lm = BigramLanguageModel(kUNK_CUTOFF, jm_lambda=args.jm_lambda,
                             add_k=args.add_k,
                             katz_cutoff=args.katz_cutoff,
                             kn_concentration=args.kn_concentration,
                             kn_discount=args.kn_discount)#first argument was kUNK_CUTOFF



    for ii in nltk.corpus.brown.sents():#for all sentences in corpus
        for jj in lm.tokenize(" ".join(ii)):#for all words in sentence, got words from sentence using tokenize
            lm.train_seen(lm._normalizer(jj))

    #while finalizing vocabulary, none of the counts are going up for words that occur more than once, adjust train_seen?

    print("Done looking at all the words, finalizing vocabulary")
    lm.finalize()
    
    
    
    #sentence = "This is a brown sample sentence"
  
    #lm.add_train(sentence)


    sentence_count = 0
    for ii in nltk.corpus.brown.sents():#for each sentence in centence corpus
        sentence_count += 1
        lm.add_train(" ".join(ii))

        if args.brown_limit > 0 and sentence_count >= args.brown_limit:
            break

    print("Trained language model with %i sentences from Brown corpus." % sentence_count)
    assert args.method in ['mle', 'add_k', \
                           'jelinek_mercer',  'laplace'], \
      "Invalid estimation method"
    
    #'kneser_ney', 'good_turing',

    sent = input()
    while sent:
        print("#".join(str(x) for x in lm.tokenize_and_censor(sent)))
        print(lm.perplexity(sent, getattr(lm, args.method)))
        sent = input()
