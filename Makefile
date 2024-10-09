.PHONY: tests doc build

lint:
	flake8 . --max-line-length=88 --exclude=doc

type-check:
	mypy mapie

tests:
	pytest -vs --doctest-modules mapie

integration-tests-v1:
	@pip install mapie --no-dependencies --target=./mapie_v1/integration_tests/mapie_v0_package >/dev/null 2>&1
	@mv ./mapie_v1/integration_tests/mapie_v0_package/mapie ./mapie_v1/integration_tests/mapie_v0_package/mapiev0
	@- export PYTHONPATH="${PYTHONPATH}:./mapie_v1/integration_tests/mapie_v0_package"; pytest -vs mapie_v1/integration_tests/tests
	@mv ./mapie_v1/integration_tests/mapie_v0_package/mapiev0 ./mapie_v1/integration_tests/mapie_v0_package/mapie

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
