.PHONY: paper site serve process

site: src/static/elm.js src/server.py src/templates/index.html

src/static/elm.min.js: src/static/elm.js
	uglifyjs $< --compress 'pure_funcs="F2,F3,F4,F5,F6,F7,F8,F9,A2,A3,A4,A5,A6,A7,A8,A9",pure_getters,keep_fargs=false,unsafe_comps,unsafe' | uglifyjs --mangle --output=$@

src/static/elm.js:
	cd src && \
		elm make Main.elm --optimize --output=static/elm.js

serve: site
	FLASK_APP=src/server.py python3 -m flask run

paper: paper.pdf

paper.pdf:
	pdflatex paper.tex

process: src/process.py
	python3 src/process.py tac_docria/en/eng.2015.train.docria glove/glove.6B.50d.txt
