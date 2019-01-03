#!/bin/bash

: >systems/predict.all.tsv
for f in predict.*.tsv; do
        (
                cat "$f"
                echo
        ) >>systems/predict.all.tsv
done
cd neleval
./scripts/run_tac16_evaluation.sh ../corpus/tac/gold/tac_kbp_2017_edl_evaluation_gold_standard_entity_mentions.tab ../systems ../out 1
cd ..
echo "Please describe the current run"
read comment

(
        echo "# $(git log -1 --pretty=oneline)"
        echo "# $comment" >>out/predict.log
        cat out/predict.all.tsv.evaluation
) >>out/predict.log

less +G out/predict.log
