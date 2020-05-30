# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = _build

# DOC_DIR = current Makefile dir
DOC_DIR=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
GENERATED_DIR=_generated
GENERATED_FILES=$(GENERATED_DIR)/classes.png $(GENERATED_DIR)/classes.pdf $(GENERATED_DIR)/packages.pdf $(GENERATED_DIR)/packages.png $(GENERATED_DIR)/recipipe_aliases.csv 

# Put it first so that "make" without argument is like "make help".
.PHONY: help
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

$(GENERATED_FILES): ../recipipe/
	mkdir -p $(GENERATED_DIR)
	# Generate recipipe_alises.csv all.
	python ./recipipe_aliases_generator.py $(GENERATED_DIR)/recipipe_aliases.csv
	# Generate class and package diagram.
	# pyreverse should be used in the root, if you use 'pyreverse ../recipipe'
	# the generated diagram is not correct (a lot of missing connections).
	cd .. && pyreverse recipipe -o png -k
	cd .. && pyreverse recipipe -o pdf -f ALL -A
	cd .. && mv classes.* $(DOC_DIR)/$(GENERATED_DIR)
	cd .. && mv packages.* $(DOC_DIR)/$(GENERATED_DIR)

generate_data: $(GENERATED_FILES)

.PHONY: clean
clean:
	rm -rf $(GENERATED_DIR)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# $(O) is meant as a shortcut for $(SPHINXOPTS).
html latexpdf epub: generate_data
	# Run autodoc on recipipe module.
	@$(SPHINXAPIDOC) -o . ../recipipe
	# Run sphinx build.
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

