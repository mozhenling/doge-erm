#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'add FISIP'

git push origin master

echo '------- update complete --------'