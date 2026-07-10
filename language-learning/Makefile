# One .txt file per language (paragraphs separated by blank lines).
# A single aligned .json file also works: make DATA=data/percy-jackson.json
DATA ?= data/percy-jackson.en.txt data/percy-jackson.zh.txt \
        data/percy-jackson.fr.txt data/percy-jackson.it.txt \
        data/percy-jackson.de.txt

.PHONY: all html epub clean
all: html epub

html:
	python3 src/build_html.py $(DATA) -o output/reader.html

epub:
	python3 src/build_epub.py $(DATA) -o output/reader.epub

clean:
	rm -f output/reader.html output/reader.epub
