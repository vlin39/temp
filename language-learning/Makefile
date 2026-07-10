DATA ?= data/percy-jackson.json

.PHONY: all html epub clean
all: html epub

html:
	python3 src/build_html.py $(DATA) -o output/reader.html

epub:
	python3 src/build_epub.py $(DATA) -o output/reader.epub

clean:
	rm -f output/reader.html output/reader.epub
