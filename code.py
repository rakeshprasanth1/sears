#!/usr/bin/env python
# coding: utf-8

# In[9]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[10]:


import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import sys


# In[13]:


# Path to sears repository
sys.path.append('sears') # noqa
import paraphrase_scorer
import onmt_model
import numpy as np


# In[14]:


ps = paraphrase_scorer.ParaphraseScorer()


# In[ ]:


import os
def load_polarity(path='/home/marcotcr/phd/datasets/sentiment-sentences/'):
    data = []
    labels = []
    f_names = ['rt-polarity.neg', 'rt-polarity.pos']
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), 'rb'):
            try:
                line.decode('utf8')
            except:
                continue
            data.append(line.decode('utf-8').strip().replace('. . .', '...'))
            labels.append(l)
    label_names = ['negative', 'positive']
    return data, labels, label_names


# In[ ]:


def load_polarity_imdb(path='/home/marcotcr/phd/datasets/sentiment-sentences-other/'):
    data = []
    labels = []
    # f_names = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
    f_names = ['imdb_labelled.txt']
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), 'rb'):
            try:
                line.decode('utf8')
            except:
                continue
            sentence, label = line.decode('utf-8').split('\t')
            label = int(label)
            data.append(sentence.strip())
            labels.append(label)
    label_names = ['negative', 'positive']
    return data, labels, label_names


# In[ ]:


import spacy
nlp = spacy.load('en')
import replace_rules
tokenizer = replace_rules.Tokenizer(nlp)


# In[ ]:


import time


# In[ ]:


import pickle
# from collections import OrderedDict


# In[ ]:


all_data = pickle.load(open('polarity.pkl', 'rb'))
# f=open('polarity.pkl', 'rb')
# all_data=OrderedDict()
# pickle.dump(all_data,'polarity.pkl', protocol=2)


# In[ ]:


data = all_data['data']
labels = all_data['labels']
label_names = all_data['label_names']
val = all_data['imdb']
val_labels = all_data['imdb_labels']


# In[ ]:


# import pickle
# pickle.dump({'data': data, 'labels': labels, 'label_names': label_names, 'imdb': val, 'imdb_labels': val_labels}, open('/tmp/polarity.pkl', 'wb'))


# In[ ]:


data = tokenizer.clean_for_model(data)


# In[ ]:


# val, val_labels, _ = load_polarity_imdb()
clean_val = tokenizer.clean_for_model(val)


# In[ ]:


import fasttext


# In[ ]:


model = fasttext.FastTextClassifier()
model.fit(data, labels, ngram_range=2, epochs=10, maxlen=100)


# In[ ]:


(model.predict(clean_val) == val_labels).mean()


# In[ ]:


val_for_onmt = [' '.join([a.text for a in x]) for x in nlp.tokenizer.pipe(val)]
val_for_onmt = [onmt_model.clean_text(x, only_upper=False) for x in val_for_onmt]


# In[ ]:


right = np.where(model.predict(clean_val) == val_labels)[0]


# In[ ]:


right_preds = np.array([val_labels[i] for i in right])


# In[ ]:


def find_flips(instance, model, topk=10, threshold=-10, ):
    orig_pred = model.predict([instance])[0]
    instance_for_onmt = onmt_model.clean_text(' '.join([x.text for x in nlp.tokenizer(instance)]), only_upper=False)
    paraphrases = ps.generate_paraphrases(instance_for_onmt, topk=topk, edit_distance_cutoff=4, threshold=threshold)
    texts = tokenizer.clean_for_model(tokenizer.clean_for_humans([x[0] for x in paraphrases]))
    preds = model.predict(texts)
    fs = [(texts[i], paraphrases[i][1]) for i in np.where(preds != orig_pred)[0]]
    return fs


# In[ ]:


import collections
orig_scores = {}
flips = collections.defaultdict(lambda: [])
for i, idx in enumerate(right):
    if i % 100 == 0:
        print(i)
    if val[idx] in flips:
        continue
    fs = find_flips(val[idx], model, topk=100, threshold=-10)
    flips[val[idx]].extend([x[0] for x in fs])


# In[ ]:


right_val = [clean_val[i] for i in right]


# In[ ]:


tr2 = replace_rules.TextToReplaceRules(nlp, right_val, [], min_freq=0.005, min_flip=0.00, ngram_size=4)


# In[ ]:


frequent_rules = []
rule_idx = {}
rule_flips = {}
for z, f in enumerate(flips):
    rules = tr2.compute_rules(f, flips[f], use_pos=True, use_tags=True)
    for rs in rules:
        for r in rs:
            if r.hash() not in rule_idx:
                i = len(rule_idx)
                rule_idx[r.hash()] = i
                rule_flips[i] = []
                frequent_rules.append(r)
            i = rule_idx[r.hash()]
            rule_flips[i].append(z)
    if z % 500 == 0:
        print (z)


# In[ ]:


token_right = tokenizer.tokenize(right_val)


# In[ ]:


model_preds = {}


# In[ ]:


len(frequent_rules)


# In[ ]:


a = time.time()
rule_flips = {}
rule_other_texts = {}
rule_other_flips = {}
rule_applies = {}
for i, r in enumerate(frequent_rules):
    idxs = list(tr2.get_rule_idxs(r))
    to_apply = [token_right[x] for x in idxs]
    applies, nt = r.apply_to_texts(to_apply, fix_apostrophe=False)
    applies = [idxs[x] for x in applies]
    old_texts = [right_val[i] for i in applies]
    old_labels = right_preds[applies]
    to_compute = [x for x in nt if x not in model_preds]
    if to_compute:
        preds = model.predict(to_compute)
        for x, y in zip(to_compute, preds):
            model_preds[x] = y
    new_labels = np.array([model_preds[x] for x in nt])
    where_flipped = np.where(new_labels != old_labels)[0]
    flips = sorted([applies[x] for x in where_flipped])
    rule_flips[i] = flips
    rule_other_texts[i] = nt
    rule_other_flips[i] = where_flipped
    rule_applies[i] = applies
    if i % 5000 == 0:
        print(i)
print(time.time() - a)


# In[ ]:


really_frequent_rules = [i for i in range(len(rule_flips)) if len(rule_flips[i]) > 1]


# In[ ]:


# to_compute_score = collections.defaultdict(lambda: set())
# for i in really_frequent_rules:
#     orig_texts =  [right_val[z] for z in rule_applies[i]]
#     new_texts = rule_other_texts[i]
#     for o, n in zip(orig_texts, new_texts):
#         to_compute_score[o].add(n)


# In[ ]:


threshold = -7.15


# In[ ]:


orig_scores = {}
for i, t in enumerate(right_val):
    orig_scores[i] = ps.score_sentences(t, [t])[0]


# I want rules s.t. the decile > -7.15. The current bottom 10% of a rule is always a lower bound on the decile, so if I see applies / 10 with score < -7.15 I can stop computing scores for that rule

# In[ ]:


ps_scores = {}


# In[ ]:


ps.last = None


# In[ ]:


rule_scores = []
rejected = set()
for idx, i in enumerate(really_frequent_rules):
    orig_texts =  [right_val[z] for z in rule_applies[i]]
    orig_scor = [orig_scores[z] for z in rule_applies[i]]
    scores = np.ones(len(orig_texts)) * -50
#     if idx in rejected:
#         rule_scores.append(scores)
#         continue
    decile = np.ceil(.1 * len(orig_texts))
    new_texts = rule_other_texts[i]
    bad_scores = 0
    for j, (o, n, orig) in enumerate(zip(orig_texts, new_texts, orig_scor)):
        if o not in ps_scores:
            ps_scores[o] = {}
        if n not in ps_scores[o]:
            if n == '':
                score = -40
            else:
                score = ps.score_sentences(o, [n])[0]
            ps_scores[o][n] = min(0, score - orig)
        scores[j] = ps_scores[o][n]
        if ps_scores[o][n] < threshold:
            bad_scores += 1
        if bad_scores >= decile:
            rejected.add(idx)
            break
    rule_scores.append(scores)
            
    if i % 100 == 0:
        print(i)


# In[ ]:


# import pickle
# pickle.dump({'ps_scores': ps_scores, 'orig_scores': orig_scores}, open('/home/marcotcr/tmp/polarity_scoresz.pkl', 'wb'))


# In[ ]:


len(rule_scores) - len(rejected)


# In[ ]:


rule_flip_scores = [rule_scores[i][rule_other_flips[really_frequent_rules[i]]] for i in range(len(rule_scores))]


# In[ ]:


frequent_flips = [np.array(rule_applies[i])[rule_other_flips[i]] for i in really_frequent_rules]


# In[ ]:


rule_precsupports = [len(rule_applies[i]) for i in really_frequent_rules]


# In[ ]:


from rule_picking import disqualify_rules
threshold=-7.15
# x = choose_rules_coverage(fake_scores, frequent_flips, frequent_supports,
disqualified = disqualify_rules(rule_scores, frequent_flips,
                          rule_precsupports, 
                      min_precision=0.0, min_flips=6, 
                         min_bad_score=threshold, max_bad_proportion=.10,
                          max_bad_sum=999999)


# In[ ]:


# [(i, x.hash()) for (i, x) in enumerate(frequent_rules) if 'text_movie -> text_film' in x.hash()]


# In[ ]:


from rule_picking import choose_rules_coverage
threshold=-7.15
a = time.time()
x = choose_rules_coverage(rule_flip_scores, frequent_flips, None,
                          None, len(right_preds),
                                frequent_scores_on_all=None, k=10, metric='max',
                      min_precision=0.0, min_flips=0, exp=True,
                         min_bad_score=threshold, max_bad_proportion=.1,
                          max_bad_sum=999999,
                         disqualified=disqualified,
                         start_from=[])
print(time.time() -a)
support_denominator = float(len(right_preds))
soup = lambda x: len(rule_applies[really_frequent_rules[x]]) / support_denominator 
prec = lambda x: frequent_flips[x].shape[0] / float(len(rule_scores[x]))
fl = len(set([a for r in x for a in frequent_flips[r]]))
print('Instances flipped: %d (%.2f)' % (fl, fl / float(len(right_preds))))
print('\n'.join(['%-5d %-5d %-5d %-35s f:%d avg_s:%.2f bad_s:%.2f bad_sum:%d Prec:%.2f Supp:%.2f' % (
                i, x[i], really_frequent_rules[r],
                frequent_rules[really_frequent_rules[r]].hash().replace('text_', '').replace('pos_', '').replace('tag_', ''),
                frequent_flips[r].shape[0],
                np.exp(rule_flip_scores[r]).mean(), (rule_scores[r] < threshold).mean(),
                (rule_scores[r] < threshold).sum(), prec(r), soup(r)) for i, r in enumerate(x)]))


# ### a couple of examples from the first rules

# In[ ]:


for r in x:
    rid = really_frequent_rules[r]
    rule =  frequent_rules[rid]
    print('Rule: %s' % rule.hash())
    print()
    for f in rule_flips[rid][:2]:
        print('%s\nP(positive):%.2f' % (right_val[f], model.predict_proba([right_val[f]])[0, 1]))
        print()
        new = rule.apply(token_right[f])[1]
        print('%s\nP(positive):%.2f' % (new, model.predict_proba([new])[0, 1]))
        print()
        print()
    print('---------------')


# In[ ]:





# In[ ]:




