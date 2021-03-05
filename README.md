# Clothes Classification

### Data

[Subset of clothing dataset](https://github.com/alexeygrigorev/clothing-dataset-small) with the top-10 most popular classes from the [full clothing dataset](https://github.com/alexeygrigorev/clothing-dataset). This is relatively small dataset (~106MB unpacked) with train/validation/test split and over 3800 images in total.

### Model

[MobileNetV2](https://arxiv.org/abs/1801.04381) was used for training due to its small size (~15MB). To be more specific, it was [Keras MobileNetV2](https://keras.io/api/applications/mobilenet/) with ImageNet weights and no top layers included. Instead of top part, few layers were added:

```python
model = tf.keras.Sequential()
model.add(mobileNet_base)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(config.num_classes, activation='softmax'))
```

Base part (extraction) of MobileNetV2 was frozen during this training. Only dense layers were trained.

### Training

![](doc/history.png)

![](doc/confusion_matrix)

### Inference

```
python inference.py --url "https://images-na.ssl-images-amazon.com/images/I/816gWcWYTuL._UL1500_.jpg"
```
> Predicted class: t-shirt (confidence score: 1.00000)

```
python inference.py --url "https://image.made-in-china.com/202f0j00yUrfpcYJYekG/Propeller-Hat-Colorful-Patchwork-Custom-Design-Cotton-Funny-Baseball-Hats.jpg"
```
> Predicted class: hat (confidence score: 0.99997)
