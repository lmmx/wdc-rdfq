---
dataset_info:
- config_name: 2015-11
  features:
  - name: subject
    dtype: string
  - name: predicate
    dtype: string
  - name: object
    dtype: string
  - name: graph
    dtype: string
  splits:
  - name: train
    num_bytes: 73462572077
    num_examples: 383925313
  download_size: 8249527680
  dataset_size: 73462572077
- config_name: 2016-10
  features:
  - name: subject
    dtype: large_string
  - name: predicate
    dtype: large_string
  - name: object
    dtype: large_string
  - name: graph
    dtype: large_string
  splits:
  - name: train
    num_bytes: 2471989472
    num_examples: 12792462
  download_size: 237471016
  dataset_size: 2471989472
- config_name: 2017-12
  features:
  - name: subject
    dtype: large_string
  - name: predicate
    dtype: large_string
  - name: object
    dtype: large_string
  - name: graph
    dtype: large_string
  splits:
  - name: train
    num_bytes: 2641496674
    num_examples: 11984942
  download_size: 301009878
  dataset_size: 2641496674
configs:
- config_name: 2015-11
  data_files:
  - split: train
    path: 2015-11/train-*
- config_name: 2016-10
  data_files:
  - split: train
    path: 2016-10/train-*
- config_name: 2017-12
  data_files:
  - split: train
    path: 2017-12/train-*
license: apache-2.0
tags:
- linked-data
pretty_name: WDC Common Crawl Embedded JSONLD
---
