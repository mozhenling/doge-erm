#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'update accepted TMM'

git push origin master

echo '------- update complete --------'