#!/bin/bash

git remote add origin http://github.com/ChiLi90/MCD19
git pull origin master

source ./bin/activate
pip install --upgrade -r requirements.txt
