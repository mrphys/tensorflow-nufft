.PHONY: lint
lint:
	pylint --rcfile=pylintrc tensorflow_nufft/python

.PHONY: docs
docs:
	rm -rf docs/_* docs/api_docs/tfft/
	$(MAKE) -C docs dirhtml
