#!/bin/bash

mkdir -p split_datasets

cp train.txt split_datasets/train.txt

cd split_datasets

for n in {0..9}; do
    echo $n
    head -1000 train.txt > train_$n.txt && sed -i '1,+999d' train.txt
done

rm train.txt

for train_set in ./*.txt; do
    echo $train_set
    python3 ../prep_data_dict.py $train_set

done

cd ..