# image_embedding_retrieval
retrieve similar images by image embedding


## 1. Extract embedding and indexing the images
```shell
python image_embedding_retrieval/extract.py --config config.json --emb --ann
```

## 2. Retrieve k similar images with given query image
```shell
python image_embedding_retrieval/retrieve.py --config config.json --k 5 --img_path ../exp/pokemon_jpg/151.jpg
```