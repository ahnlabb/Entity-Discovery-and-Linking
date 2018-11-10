.PHONY: paper site serve

site: src/static/elm.js src/server.py src/templates/index.html

src/static/elm.js:
	cd src && \
		elm make Main.elm --optimize --output=static/elm.js

serve: site
	FLASK_APP=src/server.py python3 -m flask run

paper: paper.pdf

paper.pdf:
	pdflatex paper.tex
