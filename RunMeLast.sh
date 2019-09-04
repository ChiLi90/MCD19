#!/bin/bash


git commit -a -m 'last submission'

pip freeze requirements.txt

git push -u origin master

deactivate

git remote remove origin
