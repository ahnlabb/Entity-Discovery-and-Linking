.PHONY: paper site serve process

ifneq ($(CONDA_DEFAULT_ENV),nlp-project)
$(error Not in conda environment)
endif

site: src/static/elm_debug.js src/static/elm.js src/server.py src/templates/debug.html

src/static/elm.min.js: src/static/elm.js
	uglifyjs $< --compress 'pure_funcs="F2,F3,F4,F5,F6,F7,F8,F9,A2,A3,A4,A5,A6,A7,A8,A9",pure_getters,keep_fargs=false,unsafe_comps,unsafe' | uglifyjs --mangle --output=$@

src/static/elm.js: src/Main.elm
	cd src && \
		elm make Main.elm --optimize --output=static/elm.js

src/static/elm_debug.js: src/Debug.elm
	cd src && \
		elm make Debug.elm --optimize --output=static/elm_debug.js

serve: site
	FLASK_APP=src/server.py python3 -m flask run

paper: paper.pdf

paper.pdf:
	pdflatex paper.tex

process: src/process.py
	python3 src/process.py corpus/tac/en/eng.2015.train.docria glove/glove.6B.50d.pickle
