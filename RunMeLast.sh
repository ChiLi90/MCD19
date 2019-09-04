#!/bin/bash

#git push -u origin master
git commit -a -m 'last submission'
git remote remove origin

pip freeze requirements.txt
deactivate
