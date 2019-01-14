#!/bin/bash
JAR='target/scala-2.11/Simple Project-assembly-1.0.jar'
SPARK_OPTS='--num-executors 4 --executor-cores 4 --executor-memory 1GB'
if [ -d "wiki_mappings" ]; then
  echo "Directory exists"
else
  spark-submit $SPARK_OPTS "$JAR" '../../corpus/zhwiki/part-r-*.docria'
fi
