# Probing tasks

Probing tasks are meant to analyse what linguistic information can be extracted from sentence embeddings.

## General remarks

This directory contains the probing tasks described in "[What you can cram into
a single $&!#\* vector: Probing sentence embeddings for linguistic properties.](https://arxiv.org/abs/1805.01070)"
All data sets contain 100k training instances, 10k validation instances and 10k
test instances, and in all cases they are balanced across the target classes
(in some cases, there are a few more instances, as a result of balancing
constraints in the sampling process). Each instance is on a separate line, and
contains (at least) the following tab-separated fields:

- the first field specifies the partition (tr/va/te);

- the second field specifies the ground-truth class of the instance (e.g.,
  PRES/PAST in the past_present.txt file);

- the last field contains the sentence (in space-delimited tokenized format).

In all data sets, the instances are ordered by partition, but, within each
partition, they are randomized.

See the main paper for details on how the data sets were constructed. Note,
further, that all data are post-processed to be compatible with the Moses
tokenization conventions, since the latter are assumed by the SentEval tools.

In what follows, there is a description for each of the task files.

## sentence_length.txt

This is a classification task where the goal is to predict the sentence length
which has been binned in 6 possible categories with lengths ranging in the
following intervals: --0: (5-8), 1: (9-12), 2: (13-16), 3: (17-20), 4: (21-25),
5: (26-28). These are the same bins from Adi et al. except for the two larger
ones --(30-33), (34-70)-- and lenght 29 in the last bin, all of which we
excluded because the corresponding lengths  were filtered out in our corpus
pre-processing step. This task is called SentLen in the paper.

## word_content.txt

This is a classification task with 1000 words as targets. The task is
predicting which of the target words appear on the given sentence.

We constructed the data by picking the first 1000 lower-cased words occurring
in the corpus vocabulary ordered by rank from position 2k+1 onwards, and having
length of at least 4 characters (to remove noise). Each sentence contains a
single target word, and the word occurs exactly once in the sentence.

The task is called WC in the paper.

## tree_depth.txt

This is a classification tasks where the goal is to predict the maximum depth
of the sentence's syntactic tree (with values ranging from 5 to 12).

Since sentence depth naturally correlates with sentence length, we defined a
target bivariate gaussian distribution relating sentence length and sentence
depth, set the co-variance to be diagonal, and sampled a subset of sentences to
match this distribution, obtaining a decorrelated sample.

The task is called TreeDepth in the paper.

## bigram_shift.txt

In this classification task the goal is to predict whether two consecutive
tokens within the sentence have been inverted (label I for inversion and O for
original).

The data was constructed by choosing two random consecutive tokens in the
sentence, excluding beginning of sentence and punctuation marks. We also
excluded sentences containing double quotes.

The task is called BShift in the paper.

## top_constituents.txt

This is a 20-class classification task, where the classes are given by the 19
most common top-constituent sequences in the corpus, plus a 20th category for
all other structures. The classes are:

- ADVP_NP_VP_.
- CC_ADVP_NP_VP_.
- CC_NP_VP_.
- IN_NP_VP_.
- NP_ADVP_VP_.
- NP_NP_VP_.
- NP_PP_.
- NP_VP_.
- OTHER
- PP_NP_VP_.
- RB_NP_VP_.
- SBAR_NP_VP_.
- SBAR_VP_.
- S_CC_S_.
- S_NP_VP_.
- S_VP_.
- VBD_NP_VP_.
- VP_.
- WHADVP_SQ_.
- WHNP_SQ_.

Top-constituent sequences that contained sentence-internal punctuation marks
and quotes or did not end with the . label were excluded (also from the OTHER
class).

The task is called TopConst in the paper.

## past_present.txt

This is a binary classification task, based on whether the main verb of the
sentence is marked as being in the present (PRES class) or past (PAST class)
tense. The present tense corresponds to PoS tags VBP and VBZ, whereas the past
tense corresponds to VBD.

Only sentences where the main verb has a corpus frequency of between 100 and
5,000 occurrences are considered. More importantly, a verb form can only occur
in one of the partitions. For example, the past form "provided" only occurs
in the training set.

The task is called Tense in the paper.

## subj_number.txt

Another binary classification task, this time focusing on the number of the
subject of the main clause. The classes are NN (singular) and NNS (plural or
mass: "personnel", "clientele", etc). As the class labels suggest, only common
nouns are considered.

Like above, only target noun forms with corpus frequency between 100 and 5,000
are considered, and noun forms are split across the partitions.

The task is called SubjNum in the paper.

## obj_number.txt

This binary classification task is analogous to the one above, but this time
focusing on the direct object of the main clause. The labels are again NN to
represent the singular class and NNS for the plural/mass one.

Again, only target noun forms with corpus frequency between 100 and 5,000
are considered, and noun forms are split across the partitions.

The task is called ObjNum in the paper.

## odd\_man\_out.txt

This binary task asks whether a sentence occurs as-is in the source corpus
(O label, for Original), or whether a (single) randomly picked noun or verb was
replaced with another form with the same part of speech (C label, for Changed).
The original word and the replacement have comparable frequencies (in log-scale)
for the bigrams they form with the immediately preceding and following tokens.

Both target and replacement were filtered to have corpus frequency between 40
and 400 occurrences. This range is considerably lower than for the other data
sets, because very frequent words tend to have vague meanings that are
compatible with many contexts.  More importantly, for the sentences with
replacement, the replacement words only occur in one partition. Moreover, no
sentence occurs in both the original and changed versions.

The task is called SOMO in the paper.

## coordination_inversion.txt

Binary task asking to distinguish between original sentence (class O) and
sentences where the order of two coordinated clausal conjoints has been inverted
(class I). An example of the latter is: "There was something to consider but
he might be a prince". Only sentences that contain just one coordinating
conjunction (CC) are considered, and the conjunction must coordinate two top
clauses.

In constructing the data set, we balanced the sentences by the length of the
two conjoined clauses, that is, both the original and inverted sets contain
an equal number of cases in which the first clause is longer, the second one is
longer, and they are of equal length. Also, no sentence is presented in both
original and inverted order.

The task is called CoordInv in the paper.


## References

Please considering citing [[1]](https://arxiv.org/abs/1805.01070) if using these probing tasks for analysing sentence embedding methods.

### What you can cram into a single vector: Probing sentence embeddings for linguistic properties (ACL 2018)

[1] A. Conneau, G. Kruszewski, G. Lample, L. Barrault, M. Baroni, [*What you can cram into a single vector: Probing sentence embeddings for linguistic properties*](https://arxiv.org/abs/1805.01070)

```
@article{conneau2018probing,
  title={What you can cram into a single vector: Probing sentence embeddings for linguistic properties},
  author={Conneau, Alexis and Kruszewski, German and Lample, Guillaume and Barrault, Loic and Baroni, Marco},
  journal={arXiv preprint arXiv:1805.01070},
  year={2018}
}
```

### Related work
* [X. Shi, I. Padhi, K. Knight - Does string-based neural MT learn source syntax?, EMNLP 2016](https://aclanthology.coli.uni-saarland.de/papers/D16-1159/d16-1159)
* [Y. Adi, E. Kermany, Y. Belinkov, O. Lavi, Y. Goldberg - Fine-grained analysis of sentence embeddings using auxiliary prediction tasks, ICLR 2017](https://arxiv.org/abs/1608.04207)
