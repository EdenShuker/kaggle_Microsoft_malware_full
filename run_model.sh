for i in {1..9}
do
    python extract_ngrams.py -n $i -p data/train_labels_filtered.csv
done