#!/bin/bash

source ./bin/activate
git remote add origin http://github.com/ChiLi90/MCD19
git pull origin master

pip install --upgrade -r requirements.txt
