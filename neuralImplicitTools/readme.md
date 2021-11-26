# Neural Implicit Tools
Last Modified: Jie - 08/16/2021.

**Notes**
- The cube marcher might need to be debugged. The `trainer.py` has problems training on file `data/geometry_200/3437e68d-207a-46ec-9642-e9b0461fcdaa.stl`.
- Simple tricks worked. Importance sampling is currently the best pick for us since it represents satisfying performances as well as promising robustness (important since many fancy ideas are not robust enough).
- ReLU networks appear to be able to converge fast with right initialization. We have tried `LeakyReLU` and `Sinusoidal (SIREN)`, and none is as good. Always use Kaiming Uniform initialization (instead of Xavier) on a ReLU network, use `--init he_uniform`.
- A rule of thumb is, wider networks generally can have better compatibility with larger batch sizes, which we like since larger batches enable better parallelization on GPU and faster training. Set `--damArch` to 1 to use network architecture selected by DAM (Discriminative Masking), and use larger batch size like `--batchSize 1024` or `--batchSize 2048`. The 10 layers architecture suggested by DAM is `[333, 442, 351, 340, 304, 139, 76, 53, 39, 36, 35]`.
- The `--scaler` option is preferred when the training time budget is low. However, it is not very well tuned for all the three options `StandardScaler, PowerTransformer, QuantileTransformer`. Generally speaking, `StandardScaler` is a safe choice but does not make much differences. `PowerTransformer` is usually a good choice but it can be unstable (don't use `inverse_transform` but use `_scaler.inverse_transform`). The `QuantileTransformer` can project the input distribution to a near gaussian shape, which sometimes can significantly speed up convergence. However, it is not tuned very well for us so we have to be careful when using it.
- Apart from hardware upgrades and distributed parallelism, there is room for code optimization when we can disable the debug related features to further speed up the code.

-----

## Training Presets

----
### High Budget Training
Fitting 5 million SDF points with importance sampling:

```
python trainer.py ../data/impeller/impeller.stl \
--numLayers=10 \
--outputDir ../results/impeller/ \
--epochs 6 \
--batchSize 1024 \
--learningRate 0.001 \
--num_points 5000000 \
--init he_uniform \
--damArch 1 \
--scaler StandardScaler \
--validationRes 0 \
--showVis 1 \
--reconstructionRes 256
```
Training time is around 200 ~ 250 seconds on K80.

----
### Medium Budget Training
Fitting 5 million SDF points with importance sampling:

```
python trainer.py ../data/impeller/impeller.stl \
--numLayers=10 \
--outputDir ../results/impeller/ \
--epochs 3 \
--batchSize 2048 \
--learningRate 0.001 \
--num_points 5000000 \
--init he_uniform \
--damArch 1 \
--scaler QuantileTransformer \
--validationRes 0 \
--showVis 1 \
--reconstructionRes 256
```
Training time is around 70 ~ 80 seconds on K80.
# Ran on 11/26/2021

---
### Low Budget Training
Fitting 2 million SDF points with importance sampling:

```
python trainer.py ../data/impeller/impeller.stl \
--numLayers=10 \
--outputDir ../results/impeller/ \
--epochs 4 \
--batchSize 1024 \
--learningRate 0.001 \
--num_points 2000000 \
--init he_uniform \
--damArch 1 \
--scaler PowerTransformer \
--validationRes 0 \
--showVis 1 \
--reconstructionRes 256\
```
Training time is around 50 ~ 60 seconds on K80.

---
### Very High Budget Training (Ultra Details)
Fitting 5 million SDF points with importance sampling:

```
python trainer.py ../data/impeller/impeller.stl \
--numLayers=10 \
--outputDir ../results/impeller/ \
--epochs 6 \
--batchSize 512 \
--learningRate 0.001 \
--num_points 5000000 \
--init he_uniform \
--damArch 1 \
--validationRes 0 \
--showVis 1 \
--reconstructionRes 256
```
Training time is around 350 ~ 400 seconds on K80.

### Geometry-200 Dataset (Medium Presets)
```
python trainer.py ../data/geometry_200 \
--numLayers=10 \
--outputDir ../results/geometry_200/ \
--epochs 3 \
--batchSize 2048 \
--learningRate 0.001 \
--num_points 5000000 \
--init he_uniform \
--damArch 1 \
--scaler QuantileTransformer \
--validationRes 0 \
--showVis 1 \
--reconstructionRes 256
```
### Geometry-200 Dataset (High Presets)
```
python trainer.py ../data/geometry_200 \
--numLayers=10 \
--outputDir ../results/geometry_200/ \
--epochs 6 \
--batchSize 1024 \
--learningRate 0.001 \
--num_points 5000000 \
--init he_uniform \
--damArch 1 \
--validationRes 0 \
--showVis 1 \
--reconstructionRes 256
```

### Training on the Problematic Part
```
python trainer.py ../data/geometry_200/3437e68d-207a-46ec-9642-e9b0461fcdaa.stl \
--numLayers=10 \
--outputDir ../results/geometry_200/ \
--epochs 6 \
--batchSize 1024 \
--learningRate 0.001 \
--num_points 5000000 \
--init he_uniform \
--damArch 1 \
--scaler StandardScaler \
--validationRes 0 \
--showVis 1 \
--reconstructionRes 256
```
