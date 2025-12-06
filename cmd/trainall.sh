#!/bin/bash

for dir in train_*; do
cd $dir
go build
cd ..
done

./train_mnist/train_mnist --dstmodel train_mnist/output.json.t.zlib
./train_is_virus/train_is_virus --dstmodel train_is_virus/output.json.t.zlib
./train_is_alnum/train_is_alnum --dstmodel train_is_alnum/output.json.t.zlib
./train_squareroot/train_squareroot --dstmodel train_squareroot/output.json.t.zlib
