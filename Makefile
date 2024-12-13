.PHONY: tests doc build

mapie_v0_folder_name = mapie_v0_package

lint:
	flake8 . --max-line-length=88 --exclude=doc

type-check:
	mypy mapie

v1-type-check:
	mypy mapie_v1 --exclude $(mapie_v0_folder_name)

tests:
	pytest -vs --doctest-modules mapie

integration-tests-v1:
	@pip install git+https://github.com/scikit-learn-contrib/MAPIE@master --no-dependencies --target=./mapie_v1/integration_tests/$(mapie_v0_folder_name) >/dev/null 2>&1
	@mv ./mapie_v1/integration_tests/$(mapie_v0_folder_name)/mapie ./mapie_v1/integration_tests/$(mapie_v0_folder_name)/mapiev0
	@- export PYTHONPATH="${PYTHONPATH}:./mapie_v1/integration_tests/$(mapie_v0_folder_name)"; pytest -vs mapie_v1/integration_tests/tests  -k $(pattern)
	@mv ./mapie_v1/integration_tests/$(mapie_v0_folder_name)/mapiev0 ./mapie_v1/integration_tests/$(mapie_v0_folder_name)/mapie

checks-v1-not-in-ci:
	$(MAKE) v1-type-check
	$(MAKE) integration-tests-v1 pattern=test

coverage:
	pytest -vs \
		--doctest-modules \
		--cov-branch \
		--cov=mapie \
		--cov-report term-missing \
		--pyargs mapie \
		--cov-fail-under=100 \
		--cov-config=.coveragerc

doc:
	$(MAKE) html -C doc

doctest:
	$(MAKE) doctest -C doc

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
