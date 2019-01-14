.PHONY: paper site serve process-en process-es process-zh process neleval clean-model rerun

ifneq ($(CONDA_DEFAULT_ENV),nlp-project)
$(error Not in conda environment)
endif

site: src/static/elm_debug.js src/static/elm.js src/server.py src/templates/debug.html

src/static/elm.min.js: src/static/elm.js
	uglifyjs $< --compress 'pure_funcs="F2,F3,F4,F5,F6,F7,F8,F9,A2,A3,A4,A5,A6,A7,A8,A9",pure_getters,keep_fargs=false,unsafe_comps,unsafe' | uglifyjs --mangle --output=$@

src/static/elm.js: src/Main.elm
	cd src && \
		elm make Main.elm --optimize --output=static/elm.js

src/static/elm_debug.js: src/DocDebug.elm
	cd src && \
		elm make DocDebug.elm --optimize --output=static/elm_debug.js

serve: site
	FLASK_APP=src/server.py python3 -m flask run

paper: paper.pdf

paper.pdf:
	pdflatex paper.tex

prepare-s3:
	mkdir -p corpus/tac/lang/en/ glove/
	aws s3 cp s3://nlp-project-edan70/eng.2015.train.pickle corpus/tac/lang/en/
	aws s3 cp s3://nlp-project-edan70/eng.2015.eval.pickle corpus/tac/lang/en/
	aws s3 cp s3://nlp-project-edan70/glove.6B.100d.pickle glove

process-es:
	python3 src/process.py corpus/tac/lang/es/spa.2015.eval.pickle corpus/tac/lang/es/spa.2015.train.pickle corpus/tac/lang/es/spa.2016.eval.pickle glove/tac-es-top200k.case.pickle model.es.pickle es corpus/wikimap_es.pickle --predict corpus/tac/lang/es/spa.2017.eval.pickle

process-zh:
	python3 src/process.py corpus/tac/lang/zh/cmn.2015.eval.pickle corpus/tac/lang/zh/cmn.2015.train.pickle corpus/tac/lang/zh/cmn.2016.eval.pickle glove/tac-zh.pickle model.zh.pickle zh corpus/wikimap_zh.pickle --predict corpus/tac/lang/zh/cmn.2017.eval.pickle 

process-en:
	python3 src/process.py corpus/tac/lang/en/eng.2015.eval.pickle corpus/tac/lang/en/eng.2015.train.pickle corpus/tac/lang/en/eng.2016.eval.pickle glove/glove.6B.100d.pickle model.en.pickle en corpus/wikimap_en.pickle --predict corpus/tac/lang/en/eng.2017.eval.pickle

neleval:
	./run_neleval.sh

clean-model:
	rm -f model.*.pickle

rerun: clean-model process-en process-es process-zh
