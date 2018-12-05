#!/bin/bash
sed 's/\[//g' y_test.csv > temp1
sed 's/ //g' temp1 > temp2
sed 's/]//g' temp2 > y_test.csv
rm -f temp1 temp2
