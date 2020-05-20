# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	# Generate recipipe_alises.csv all.
	python ./recipipe_aliases_generator.py
	# Generate class and package diagram.
	pyreverse ../recipipe -o png -k
	pyreverse ../recipipe -o pdf -f ALL -A
	mv classes.* _generated/
	mv packages.* _generated/
	# Run autodoc on recipipe module.
	@$(SPHINXAPIDOC) -o . ../recipipe
	# Run sphinx build.
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
