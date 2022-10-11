#!/bin/bash

mkdir -p compiled images

# ############ Convert friendly and compile to openfst ############
for i in friendly/*.txt; do
    echo "Converting friendly: $i"
    python3 compact2fst.py  $i  > sources/$(basename $i ".formatoAmigo.txt").txt
done


# ############ convert words to openfst ############
for w in tests/*.str; do
	echo "Converting words: $w"
	python3 word2fst.py `cat $w` > tests/$(basename $w ".str").txt
done


# ############ Compile source transducers ############
for i in sources/*.txt tests/*.txt; do
	echo "Compiling: $i"
    fstcompile --isymbols=syms.txt --osymbols=syms.txt $i | fstarcsort > compiled/$(basename $i ".txt").fst
done

# ############ CORE OF THE PROJECT  ############

# #### COMPOSING FSTs ####

cp compiled/step1.fst compiled/allsteps.fst
for step in {2..9}; do
    echo $step
    fstcompose compiled/allsteps.fst compiled/step$step.fst compiled/allsteps.fst
done

# #### CREATING INVERSION ####

fstinvert compiled/allsteps.fst compiled/allsteps-invert.fst


# ############ tests  ############

echo "Testing"

for w in compiled/t-*.fst; do
    fstcompose $w compiled/allsteps.fst | fstshortestpath | fstproject --project_type=output |
    fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=./syms.txt
done
