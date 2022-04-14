#!/bin/sh
conda activate adverb
CUDA_VISIBLE_DEVICES=3, python webservice.py 
$SHELL