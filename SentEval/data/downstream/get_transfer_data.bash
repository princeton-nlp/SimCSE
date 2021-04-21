# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Download and tokenize data with MOSES tokenizer
#

data_path=.
preprocess_exec=./tokenizer.sed

# Get MOSES
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git
SCRIPTS=mosesdecoder/scripts
MTOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LOWER=$SCRIPTS/tokenizer/lowercase.perl

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

PTBTOKENIZER="sed -f tokenizer.sed"

mkdir $data_path

TREC='http://cogcomp.cs.illinois.edu/Data/QA/QC'
SICK='http://alt.qcri.org/semeval2014/task1/data/uploads'
BINCLASSIF='https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip'
SSTbin='https://raw.githubusercontent.com/PrincetonML/SIF/master/data'
SSTfine='https://raw.githubusercontent.com/AcademiaSinicaNLPLab/sentiment_dataset/master/data/'
STSBenchmark='http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz'
SNLI='https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
MULTINLI='https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip'
COCO='https://dl.fbaipublicfiles.com/senteval/coco_r101_feat'

# MRPC is a special case (we use "cabextract" to extract the msi file on Linux, see below)
MRPC='https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi'

# STS 2012, 2013, 2014, 2015, 2016
declare -A STS_tasks
declare -A STS_paths
declare -A STS_subdirs

STS_tasks=(["STS12"]="MSRpar MSRvid SMTeuroparl surprise.OnWN surprise.SMTnews" ["STS13"]="FNWN headlines OnWN" ["STS14"]="deft-forum deft-news headlines OnWN images tweet-news" ["STS15"]="answers-forums answers-students belief headlines images" ["STS16"]="answer-answer headlines plagiarism postediting question-question")

STS_paths=(["STS12"]="http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip" ["STS13"]="http://ixa2.si.ehu.es/stswiki/images/2/2f/STS2013-en-test.zip" ["STS14"]="http://ixa2.si.ehu.es/stswiki/images/8/8c/STS2014-en-test.zip" ["STS15"]="http://ixa2.si.ehu.es/stswiki/images/d/da/STS2015-en-test.zip"
["STS16"]="http://ixa2.si.ehu.es/stswiki/images/9/98/STS2016-en-test.zip")

STS_subdirs=(["STS12"]="test-gold" ["STS13"]="test-gs" ["STS14"]="sts-en-test-gs-2014" ["STS15"]="test_evaluation_task2a" ["STS16"]="sts2016-english-with-gs-v1.0")




### Get Stanford Sentiment Treebank (SST) binary classification task
# SST binary
mkdir -p $data_path/SST/binary
for split in train dev test
do
    curl -Lo $data_path/SST/binary/sentiment-$split $SSTbin/sentiment-$split
done

# SST fine-grained
mkdir -p $data_path/SST/fine/
for split in train dev test
do
  curl -Lo $data_path/SST/fine/sentiment-$split $SSTfine/stsa.fine.$split
done

### STS datasets

# STS12, STS13, STS14, STS15, STS16
mkdir $data_path/STS

for task in "${!STS_tasks[@]}"; #"${!STS_tasks[@]}";
do
    fpath=${STS_paths[$task]}
    echo $fpath
    curl -Lo $data_path/STS/data_$task.zip $fpath
    unzip $data_path/STS/data_$task.zip -d $data_path/STS
    mv $data_path/STS/${STS_subdirs[$task]} $data_path/STS/$task-en-test
    rm $data_path/STS/data_$task.zip

    for sts_task in ${STS_tasks[$task]}
    do
        fname=STS.input.$sts_task.txt
        task_path=$data_path/STS/$task-en-test/

        if [ "$task" = "STS16" ] ; then
            echo 'Handling STS2016'
            mv $task_path/STS2016.input.$sts_task.txt $task_path/$fname
            mv $task_path/STS2016.gs.$sts_task.txt $task_path/STS.gs.$sts_task.txt
        fi



        cut -f1 $task_path/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $task_path/tmp1
        cut -f2 $task_path/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $task_path/tmp2
        paste $task_path/tmp1 $task_path/tmp2 > $task_path/$fname
        rm $task_path/tmp1 $task_path/tmp2
    done

done


# STSBenchmark (http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)

curl -Lo $data_path/Stsbenchmark.tar.gz $STSBenchmark
tar -zxvf $data_path/Stsbenchmark.tar.gz -C $data_path
rm $data_path/Stsbenchmark.tar.gz
mv $data_path/stsbenchmark $data_path/STS/STSBenchmark

for split in train dev test
do
    fname=sts-$split.csv
    fdir=$data_path/STS/STSBenchmark
    cut -f1,2,3,4,5 $fdir/$fname > $fdir/tmp1
    cut -f6 $fdir/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $fdir/tmp2
    cut -f7 $fdir/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $fdir/tmp3
    paste $fdir/tmp1 $fdir/tmp2 $fdir/tmp3 > $fdir/$fname
    rm $fdir/tmp1 $fdir/tmp2 $fdir/tmp3
done




### download TREC
mkdir $data_path/TREC

for split in train_5500 TREC_10
do
    urlname=$TREC/$split.label
    curl -Lo $data_path/TREC/$split.label $urlname
    sed -i -e "s/\`//g" $data_path/TREC/$split.label
    sed -i -e "s/'//g" $data_path/TREC/$split.label
done




### download SICK
mkdir $data_path/SICK

for split in train trial test_annotated
do
    urlname=$SICK/sick_$split.zip
    curl -Lo $data_path/SICK/sick_$split.zip $urlname
    unzip $data_path/SICK/sick_$split.zip -d $data_path/SICK/
    rm $data_path/SICK/readme.txt
    rm $data_path/SICK/sick_$split.zip
done

for split in train trial test_annotated
do
    fname=$data_path/SICK/SICK_$split.txt
    cut -f1 $fname | sed '1d' > $data_path/SICK/tmp1
    cut -f4,5 $fname | sed '1d' > $data_path/SICK/tmp45
    cut -f2 $fname | sed '1d' | $MTOKENIZER -threads 8 -l en -no-escape > $data_path/SICK/tmp2
    cut -f3 $fname | sed '1d' | $MTOKENIZER -threads 8 -l en -no-escape > $data_path/SICK/tmp3
    head -n 1 $fname > $data_path/SICK/tmp0
    paste $data_path/SICK/tmp1 $data_path/SICK/tmp2 $data_path/SICK/tmp3 $data_path/SICK/tmp45 >> $data_path/SICK/tmp0
    mv $data_path/SICK/tmp0 $fname
    rm $data_path/SICK/tmp*
done





### download MR CR SUBJ MPQA
# Download and unzip file
curl -Lo $data_path/data_classif.zip $BINCLASSIF
unzip $data_path/data_classif.zip -d $data_path/data_bin_classif
rm $data_path/data_classif.zip

# MR
mkdir $data_path/MR
cat -v $data_path/data_bin_classif/data/rt10662/rt-polarity.pos | $PTBTOKENIZER > $data_path/MR/rt-polarity.pos
cat -v $data_path/data_bin_classif/data/rt10662/rt-polarity.neg | $PTBTOKENIZER > $data_path/MR/rt-polarity.neg

# CR
mkdir $data_path/CR
cat -v $data_path/data_bin_classif/data/customerr/custrev.pos | $PTBTOKENIZER > $data_path/CR/custrev.pos
cat -v $data_path/data_bin_classif/data/customerr/custrev.neg | $PTBTOKENIZER > $data_path/CR/custrev.neg

# SUBJ
mkdir $data_path/SUBJ
cat -v $data_path/data_bin_classif/data/subj/subj.subjective | $PTBTOKENIZER > $data_path/SUBJ/subj.subjective
cat -v $data_path/data_bin_classif/data/subj/subj.objective | $PTBTOKENIZER > $data_path/SUBJ/subj.objective

# MPQA
mkdir $data_path/MPQA
cat -v $data_path/data_bin_classif/data/mpqa/mpqa.pos | $PTBTOKENIZER > $data_path/MPQA/mpqa.pos
cat -v $data_path/data_bin_classif/data/mpqa/mpqa.neg | $PTBTOKENIZER > $data_path/MPQA/mpqa.neg

# CLEAN-UP
rm -r $data_path/data_bin_classif

### download SNLI
mkdir $data_path/SNLI
curl -Lo $data_path/SNLI/snli_1.0.zip $SNLI
unzip $data_path/SNLI/snli_1.0.zip -d $data_path/SNLI
rm $data_path/SNLI/snli_1.0.zip
rm -r $data_path/SNLI/__MACOSX

for split in train dev test
do
    fpath=$data_path/SNLI/$split.snli.txt
    awk '{ if ( $1 != "-" ) { print $0; } }' $data_path/SNLI/snli_1.0/snli_1.0_$split.txt | cut -f 1,6,7 | sed '1d' > $fpath
    cut -f1 $fpath > $data_path/SNLI/labels.$split
    cut -f2 $fpath | $PTBTOKENIZER > $data_path/SNLI/s1.$split
    cut -f3 $fpath | $PTBTOKENIZER > $data_path/SNLI/s2.$split
    rm $fpath
done
rm -r $data_path/SNLI/snli_1.0




### Get COCO captions and resnet-101 2048d-features
# Captions : Copyright (c) 2015, COCO Consortium. All rights reserved.
mkdir $data_path/COCO
for split in train valid test
do
    curl -Lo $data_path/COCO/$split.pkl $COCO/$split.pkl
done




### download MRPC
mkdir $data_path/MRPC
curl -Lo $data_path/MRPC/msr_paraphrase_train.txt https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt
curl -Lo $data_path/MRPC/msr_paraphrase_test.txt https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt


# remove moses folder
rm -rf mosesdecoder
