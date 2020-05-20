# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = _build

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
	pyreverse ../recipipe -o png -k
	pyreverse ../recipipe -o pdf -f ALL -A
	mv classes.* $(GENERATED_DIR)
	mv packages.* $(GENERATED_DIR)

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

