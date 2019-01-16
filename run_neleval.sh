#!/bin/bash

REPLY='y'

if ! git diff-index --quiet HEAD --; then
        git diff --compact-summary HEAD
        echo
        read -p "The index is not clean. Do you want to continue? (y/n) " -n 1 -r
        echo
fi
if [[ $REPLY =~ ^[Yy]$ ]]; then
        outfile="systems/predict.all.tsv"
        (
                for f in predict.*.tsv; do
                        cat "$f"
                        echo
                done
        ) >"$outfile"
        cd neleval
        ./scripts/run_tac16_evaluation.sh ../corpus/tac/gold/tac_kbp_2017_edl_evaluation_gold_standard_entity_mentions.tab ../systems ../out 1
        cd ..
        echo "Please describe the current run"
        read comment

        (
        echo "# $(git log -1 --pretty="format:%h %s %d")"

        echo "# $comment" >>out/predict.log
        cat out/predict.all.tsv.evaluation
        ) >>out/predict.log

        less +G out/predict.log
fi
