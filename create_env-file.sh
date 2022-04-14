#!/bin/sh
conda env export --no-history --no-builds --name adverb | grep -v "prefix" > environment.yml