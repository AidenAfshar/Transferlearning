import tensorflow.keras.models as models
import tensorflow.keras.utils as utils
import tensorflow as tf
from random import randrange, choice
import pickle

def predictionMax(L):
    # Gets maximum prediction from list of lists containing predictions
    prediction = [0,0]
    for i in L:
        if i[0] > prediction[0]: # Compares confidence
            prediction=i
    return prediction[1]

def main():
    showConfidence = input("Show Confidence percentages for each trial? [Y/N] or [Q] to exit: ").lower()
    while showConfidence != "y" and showConfidence != "n" and showConfidence != "q":
        showConfidence = input("Try again. Please input [Y/N] to continue or [Q] to exit: ")
    if showConfidence == "q":
        return
    print("Please wait...\nLoading Model...")

    load_path = 'saves/model_save'
    model = models.load_model(load_path)
    with open(f'{load_path}/class_names.data', 'rb') as file:
        class_names = pickle.load(file)

    correct = 0
    num_test_images = input("How many images do you want to test? [Enter int]: ")
    while True:
        try:
            num_test_images = int(num_test_images)
            if num_test_images < 1:
                num_test_images = input("Please enter an integer above 0: ")
            else:
                break
        except:
            num_test_images = input("Please enter an integer above 0: ")
            
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
        for i in range(len(predictions)):
            predictions[i] = [predictions[i], class_names[i]]
        maxPrediction = predictionMax(predictions)
        print(f"------------------------\nOriginal: {butterfly}")
        print(f"Prediction : {maxPrediction}")
        if showConfidence == "y":
            for prediction in predictions:
                print(f"Class: {prediction[1]:>25} Confidence: {prediction[0] *100:.2f}%")
        if butterfly == maxPrediction:
            correct += 1

    print(f"Number of test images: {num_test_images}, Correct: {correct}, % correct: {correct/num_test_images * 100:.0f}%")
    return

main()
