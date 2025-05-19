#!/usr/bin/env bash

python3 parser/parser.py && python3 setup.py develop && mv _pyift.*.so pyift/

# pipenv run python3 parser/parser.py && pipenv run pip install -e ./ && mv _pyift.*.so pyift/
