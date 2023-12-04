#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'initial commit'

git push origin master

echo '------- update complete --------'