import tensorflow.keras.models as models
import tensorflow.keras.utils as utils
import tensorflow as tf
from subprocess import run
from random import randrange, choice
import pickle

load_path = 'saves/model_save'
model = models.load_model(load_path)

with open(f'{load_path}/class_names.data', 'rb') as file:
    class_names = pickle.load(file)

print("class Names: ")
print(class_names)

correct = 0
num_test_images = 50
for i in range(num_test_images):
    butterfly = choice(class_names) # Chooses a random item from the list of class names
    # Chooses random image to test (there are a bit over 100 images for each class, so 100 is the limit)
    img_num = randrange(1,100)
    img = utils.load_img(
        f'butterflies/{butterfly}/{img_num:03d}.jpg',
        target_size = (224, 224),
    )
    img_array = utils.img_to_array(img)
    img_array = img_array[tf.newaxis, ...]
    predictions = model(img_array).numpy()
    predictions = list(predictions[0])
    prediction = predictions.index(max(predictions))
    print(f"{butterfly} {class_names[prediction]}")
    if butterfly == class_names[prediction]:
        correct += 1

print(f"Number of test images: {num_test_images}, Correct: {correct}, % correct: {correct/num_test_images * 100:.0f}%")
