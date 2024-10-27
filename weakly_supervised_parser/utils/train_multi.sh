#!/bin/sh

git checkout main

./train.sh

git checkout spmrl-german

./train.sh
