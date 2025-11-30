# Makefile for building the book with Pandoc

# Output directory
BUILD_DIR := build

# Metadata file
META := metadata.yaml

# Front matter files (ordered)
FRONT_MATTER := \
	chapters/title-page.md \
	chapters/copyright.md \
	chapters/dedication.md \
	chapters/disclaimers.md \
	chapters/about-author.md \
	chapters/preface.md \
	chapters/how-to-use.md \
	chapters/conventions.md \
	chapters/acknowledgements.md

# Back matter files (ordered)
BACK_MATTER := \
	chapters/back-matter.md

# All numbered chapter files, sorted numerically (1,2,3,...10,11,...)
CHAPTERS := $(shell printf '%s\n' chapters/chapter-*.md | sort -V)

# All sources in reading order
SOURCES := $(FRONT_MATTER) $(CHAPTERS) $(BACK_MATTER)

# Font configuration (override from command line if you like)
# Examples:
#   make pdf MAINFONT="TeX Gyre Pagella" SANSFONT="TeX Gyre Heros" MONOFONT="TeX Gyre Cursor"
#   make pdf MAINFONT="Libertinus Serif" SANSFONT="Libertinus Sans" MONOFONT="Fira Code"
MAINFONT ?= "Latin Modern Roman"
SANSFONT ?= "Latin Modern Sans"
MONOFONT ?= "Latin Modern Mono"

# Common Pandoc options
PANDOC_OPTS := \
	--toc \
	--toc-depth=2 \
	--metadata-file=$(META) \
	--pdf-engine=/usr/bin/xelatex \
	-V mainfont=$(MAINFONT) \
	-V sansfont=$(SANSFONT) \
	-V monofont=$(MONOFONT) \
	-V fontsize=11pt \
	-V linestretch=1.2 \
	-V geometry:margin=1in \
	-V colorlinks=true \
	-V linkcolor=blue \
	-V urlcolor=blue \
	-V toccolor=gray \
	--highlight-style=tango

# Optional Pandoc template (e.g. 'eisvogel' if installed in your pandoc templates dir)
# Usage:
#   make pdf TEMPLATE=eisvogel
#   make epub TEMPLATE=
TEMPLATE ?=
PANDOC_TEMPLATE_OPT := $(if $(TEMPLATE),--template=$(TEMPLATE),)

.PHONY: all clean epub html pdf check

all: epub html

# Show which files will be included
check:
	@echo "Building from these sources (in order):"
	@printf '  %s\n' $(SOURCES)

# Build EPUB (good for Kindle KDP upload)
epub: $(BUILD_DIR)/book.epub

$(BUILD_DIR)/book.epub: $(SOURCES) $(META)
	mkdir -p $(BUILD_DIR)
	pandoc $(SOURCES) $(PANDOC_OPTS) $(PANDOC_TEMPLATE_OPT) -o $@

# Simple HTML version (nice for quick preview)
html: $(BUILD_DIR)/book.html

$(BUILD_DIR)/book.html: $(SOURCES) $(META)
	mkdir -p $(BUILD_DIR)
	pandoc $(SOURCES) $(PANDOC_OPTS) $(PANDOC_TEMPLATE_OPT) -s -o $@

# Optional: PDF via LaTeX (requires texlive + xelatex)
pdf: $(BUILD_DIR)/book.pdf

$(BUILD_DIR)/book.pdf: $(SOURCES) $(META)
	mkdir -p $(BUILD_DIR)
	pandoc $(SOURCES) $(PANDOC_OPTS) $(PANDOC_TEMPLATE_OPT) -o $@

clean:
	rm -rf $(BUILD_DIR)
