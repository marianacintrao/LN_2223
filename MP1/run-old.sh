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








# ############ generate PDFs  ############
echo "Starting to generate PDFs"
for i in compiled/step*.fst; do
	echo "Creating image: images/$(basename $i '.fst').pdf"
   fstdraw --portrait --isymbols=syms.txt --osymbols=syms.txt $i | dot -Tpdf > images/$(basename $i '.fst').pdf
done



# ############ tests  ############

echo "Testing ABCDE"


fstcompose compiled/t-word1.fst compiled/step1.fst | fstshortestpath | fstproject --project_type=output |
fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=./syms.txt

