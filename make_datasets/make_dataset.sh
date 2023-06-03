#! /bin/bash

echo '*** Be Patient! The Whole Process May Take Several Hours! ***'

echo 'Creating folders to keep all data...'
mkdir ../data
mkdir ../data/wikipedia

echo 'Downloading Wikipedia dump...'
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
echo 'Use WikiExtractor to get the json format of Wikipedia articles...'
python -m wikiextractor.WikiExtractor --json -l enwiki-latest-pages-articles.xml --no-template

echo 'Running MongoDB...'
sudo docker compose up -d
echo 'Import Wikipedia pages to MongoDB...'
python import_dump_to_mongo.py
echo 'Extracting text anchors in Wikipedia pages and store the results in MongoDB...'
python add_anchors_to_mongo.py
echo 'Extract the id of pages which contain a context related to a Wikipedia page...'
python add_context_ids_to_mongo.py
echo 'Adding Wikidata Information to MongoDB...'
python add_wikidata_to_mongo.py

echo 'Export MongoDB data to a JsonL file...'
python mongo_to_jsonl.py
echo 'Splitting data to three train/test/validation splits...'
python split_data.py
echo 'Convert data splits in JsonL format to csv files...'
python make_csv_data.py

