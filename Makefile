# Makefile for building the book with Pandoc

# Output directory
BUILD_DIR := build

# Metadata file
META := metadata.yaml

# All chapter files, sorted numerically (1,2,3,...10,11,...)
CHAPTERS := $(shell printf '%s\n' chapters/chapter-*.md | sort -V)

# Common Pandoc options
PANDOC_OPTS := --toc --toc-depth=2 --metadata-file=$(META) --pdf-engine=/usr/bin/xelatex

.PHONY: all clean epub html pdf

all: epub html

# Build EPUB (good for Kindle KDP upload)
epub: $(BUILD_DIR)/book.epub

$(BUILD_DIR)/book.epub: $(CHAPTERS) $(META)
	mkdir -p $(BUILD_DIR)
	pandoc $(CHAPTERS) $(PANDOC_OPTS) -o $@

# Simple HTML version (nice for quick preview)
html: $(BUILD_DIR)/book.html

$(BUILD_DIR)/book.html: $(CHAPTERS) $(META)
	mkdir -p $(BUILD_DIR)
	pandoc $(CHAPTERS) $(PANDOC_OPTS) -o $@

# Optional: PDF via LaTeX (requires texlive)
pdf: $(BUILD_DIR)/book.pdf

$(BUILD_DIR)/book.pdf: $(CHAPTERS) $(META)
	mkdir -p $(BUILD_DIR)
	pandoc $(CHAPTERS) $(PANDOC_OPTS) -o $@

clean:
	rm -rf $(BUILD_DIR)
