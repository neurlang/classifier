#!/bin/bash

for dir in infer_*; do
cd $dir
go build
cd ..
done

./infer_mnist/infer_mnist --resume --dstmodel train_mnist/output.78.json.t.zlib
./infer_is_virus/infer_is_virus --resume --dstmodel train_is_virus/output.json.t.zlib
./infer_is_alnum/infer_is_alnum --resume --dstmodel train_is_alnum/output.json.t.zlib
./infer_squareroot/infer_squareroot --resume --dstmodel train_squareroot/output.json.t.zlib
