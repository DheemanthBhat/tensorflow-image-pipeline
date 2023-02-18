# TensorFlow Image Pipeline

## StackOverflow Question: Why shape of 3D tensor (of an image) in `filter()` has None?

I am following [this tutorial][1] on [tensorflow.org][2].

I have folder _images_ with two folders _cat_ and _dog_ in it. Following above tutorial I am trying to convert .jpg and .png images to features (NumPy array) for modeling.

### Problem

After processing the images to tensors I found that some images were converted to tensor with shape `(28, 28, 4)`. So I added condition to filter out such tensors. This logic works while explicitly looping each tensor, using `for` loop, after converting it to numpy array, but same logic does not work when used with `filter`.

Please help me fix this `filter()` I went through [`filter()`][3] documentation and could not find any solution.

### Source code

```python
import tensorflow as tf
import os

print("TensorFlow version:", tf.__version__)

def process_image(file_path_tensor):
    parts = tf.strings.split(file_path_tensor, os.sep)
    label = parts[-2]

    image = tf.io.read_file(file_path_tensor)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [128, 128])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255

    return image, label


def check_shape(x, y):
    print("\nShape received in filter():", x.shape)
    d1, d2, d3 = x.shape
    return d3 == 3


images_ds = tf.data.Dataset.list_files("./images/*/*", shuffle=True)

file_path = next(iter(images_ds))
image, label = process_image(file_path)

print("Shape:", image.shape)
print("Class label:", label.numpy().decode())

# ETL pipeline.
X_y_tensors = (
    images_ds
    .map(process_image)   # Extra and Transform
    .filter(check_shape)  # Filter
    .as_numpy_iterator()  # Load
)

print("\nTechnique 1:")
print("Final X count:", len(list(X_y_tensors)))


X_y_tensors = images_ds.map(process_image)

count = 0
for x, y in X_y_tensors:
    d1, d2, d3 = x.shape
    if d3 > 3:
        continue
    count += 1

print("\nTechnique 1:")
print("Final X count:", count)
```

### Output
```
TensorFlow version: 2.6.0
Shape: (128, 128, 3)
Class label: cat

Shape received in filter(): (128, 128, None)

Technique 1:
Final X count: 0

Technique 1:
Final X count: 123
```

As it can be seen,

1. Count is 0 when _Technique 1_ is used to filter tensors, since the shape of the tensor received is `(128, 128, None)`.
1. Count is 123 (image count after filtering) when _Technique 2_ is used.

I do not think [this][5] is an issue since I am **not using batches**.

[Full code with dataset][4]

[1]: https://www.tensorflow.org/guide/data#preprocessing_data
[2]: https://www.tensorflow.org/
[3]: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#filter
[4]: https://github.com/DheemanthBhat/tensorflow-image-pipeline
[5]: https://stackoverflow.com/questions/58331837/filter-data-in-tensorflow