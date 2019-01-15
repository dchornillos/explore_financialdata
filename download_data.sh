#!/bin/bash 
# Script to download and unzip the original data from http://sorry.vse.cz/~berka/challenge/pkdd1999/

mkdir data_berka
cd data_berka
wget http://sorry.vse.cz/~berka/challenge/pkdd1999/data_berka.zip
unzip data_berka.zip
rm data_berka.zip
cd ..
