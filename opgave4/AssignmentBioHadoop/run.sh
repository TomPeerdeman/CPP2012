#!/bin/bash 

HADOOP_HOME=~/hadoop/
BASE_DIR=$HOME/CPP2012/opgave4/AssignmentBioHadoop/
NUM_OF_REC=47550 
NUM_OF_SPLIT=$1
REC_PER_SPLIT=$(( $NUM_OF_REC / $NUM_OF_SPLIT )) 
echo $REC_PER_SPLIT
DATE=`date +%s`

time $HADOOP_HOME/bin/hadoop  jar $BASE_DIR/nl.uva.AssignmentMapreduce.jar data/query.fa data/fasta/ data/BLOSUM100 10 data/out-$DATE -recordsPerSplit $REC_PER_SPLIT
