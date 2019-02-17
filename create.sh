#!/bin/bash

if [ "$#" -ne 1 ]; 
then
 echo "sh create.sh projectname"
else
 rm -r build
 mkdir build
 cd build
 cmake ..
 make 
 mv $1 ..
 cd ..
fi


