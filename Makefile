### Config ###

.PHONY: tests doc build
integration_tests_folder_name = tests_v1/integration_tests
mapie_v0_folder_name = mapie_v0_package


### Checks that are run in GitHub CI ###

lint:
	flake8 . --max-line-length=88 --exclude=doc

type-check:
	mypy mapie

coverage:
	# We may need to add the v1 test suite here if we remove some v0 tests, to keep a 100% coverage
	pytest -vsx \
		--cov-branch \
		--cov=mapie \
		--cov-report term-missing \
		--pyargs mapie \
		--cov-fail-under=100 \
		--cov-config=.coveragerc \
		--no-cov-on-fail

v1-tests:
	# To replace with v1-coverage when we reach 100%
	python -m pytest -vs tests_v1 --ignore=$(integration_tests_folder_name)

v1-docstring-tests:
	pytest -vs --doctest-modules mapie_v1 --ignore=$(integration_tests_folder_name)


### Checks that are run in ReadTheDocs CI ###

doc:
	$(MAKE) html -C doc

doctest:
	# Seems to catch more errors than pytest --doctest-modules flag
	$(MAKE) doctest -C doc


### Local utilities ###

tests:
	pytest -vs --doctest-modules mapie

clean-doc:
	$(MAKE) clean -C doc

build:
	python setup.py sdist bdist_wheel

clean-build:
	rm -rf build dist MAPIE.egg-info

clean:
	rm -rf .mypy_cache .pytest_cache .coverage*
	rm -rf **__pycache__
	$(MAKE) clean-build
	$(MAKE) clean-doc


### Local utilities (v1 specific) ###

v1-all-checks:
	$(MAKE) v1-checks-not-in-ci
	$(MAKE) v1-tests
	$(MAKE) v1-docstring-tests

v1-checks-not-in-ci:
	$(MAKE) v1-type-check # Issues when trying to include it in CI, see task "Dépendances v1 + MyPy stricter in CI" in project board
	$(MAKE) v1-integration-tests # We don't want to include this in CI, will be removed at v1 release

v1-type-check:
	mypy mapie_v1 --disallow-untyped-defs --exclude $(mapie_v0_folder_name)

v1-integration-tests:
	# Run `make v1-integration-tests params="-m classification"` to select only classification tests (for example)
	# Delete the mapie v0 folder to reinstall the latest version (master)
	@if [ ! -d "./$(integration_tests_folder_name)/$(mapie_v0_folder_name)" ]; then pip install git+https://github.com/scikit-learn-contrib/MAPIE@master --no-dependencies --target=./$(integration_tests_folder_name)/$(mapie_v0_folder_name) >/dev/null 2>&1; fi
	@mv ./$(integration_tests_folder_name)/$(mapie_v0_folder_name)/mapie ./$(integration_tests_folder_name)/$(mapie_v0_folder_name)/mapiev0
	@- export PYTHONPATH="${PYTHONPATH}:./$(integration_tests_folder_name)/$(mapie_v0_folder_name)"; pytest -vs $(integration_tests_folder_name)/tests $(params)
	@mv ./$(integration_tests_folder_name)/$(mapie_v0_folder_name)/mapiev0 ./$(integration_tests_folder_name)/$(mapie_v0_folder_name)/mapie

v1-coverage:
	# To add in CI when we reach 100%
	pytest -vsx \
		--cov-branch \
		--cov=mapie_v1 \
		--cov-report term-missing \
		--pyargs tests_v1 --ignore=$(integration_tests_folder_name) \
		--cov-fail-under=100 \
		--no-cov-on-fail
