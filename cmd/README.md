# Downloading pretrained models

Download the models from: [https://www.hashtron.cloud/dl/classifier_models_v0.1/](SERVER). The largest models have 2MB.

```bash
wget https://www.hashtron.cloud/dl/classifier_models_v0.1/infer_mnist.78.json.t.zlib -O train_mnist/output.78.json.t.zlib
wget https://www.hashtron.cloud/dl/classifier_models_v0.1/infer_is_virus.100.json.t.zlib -O train_is_virus/output.100.json.t.zlib
wget https://www.hashtron.cloud/dl/classifier_models_v0.1/infer_is_alnum.100.json.t.zlib -O train_is_alnum/output.100.json.t.zlib
wget https://www.hashtron.cloud/dl/classifier_models_v0.1/infer_squareroot.100.json.t.zlib -O train_squareroot/output.100.json.t.zlib
```

# Running the models

Compile the inference

```bash
for dir in infer_*; do
cd $dir
go build
cd ..
done
```

Run the inference

```bash
./infer_mnist/infer_mnist --resume --dstmodel train_mnist/output.78.json.t.zlib
./infer_is_virus/infer_is_virus --resume --dstmodel train_is_virus/output.100.json.t.zlib
./infer_is_alnum/infer_is_alnum --resume --dstmodel train_is_alnum/output.100.json.t.zlib
./infer_squareroot/infer_squareroot --resume --dstmodel train_squareroot/output.100.json.t.zlib
```

# Training yourself (optional)

Compile the training

```bash
for dir in train_*; do
cd $dir
go build
cd ..
done
```

Run training

```bash
./train_mnist/train_mnist
./train_is_virus/train_is_virus
./train_is_alnum/train_is_alnum
./train_squareroot/train_squareroot
```


