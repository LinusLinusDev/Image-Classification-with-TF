# Group Project
# Group Name: Aachen represent
# Name: Linus Palm
# Date: 05/03/2023
from keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog


def main():
    # Load model
    saved_model_path = 'models/gender_detection_10000.h5'
    model = load_model(saved_model_path)

    # Create window
    root = Tk()
    root.title('Gender detection')
    root.geometry("800x600+100+100")

    # Function to open and classify the image
    def open_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            # open and classify image
            img = cv2.imread(file_path)
            classify_image(img)

            # open and show image
            image = Image.open(file_path)
            image = image.resize((356, 436))
            test = ImageTk.PhotoImage(image)
            label.configure(image=test)
            label.image = test

    # Function to classify the image using our model
    def classify_image(img):
        # preprocess picture
        img = tf.image.resize(img, (256, 256))
        img = np.expand_dims(img / 255, 0)

        # predict gender
        prediction = model.predict(img)

        # show result
        if prediction > 0.5:
            label_text = 'Result: Male'
        else:
            label_text = 'Result: Female'
        result_label.configure(text=label_text)

    # create button to select picture
    open_button = Button(root, text="Select image", command=open_image)
    open_button.pack()

    # create label to show picture
    label = Label(root)
    label.pack()

    # create label to show result
    result_label = Label(root, font=('Arial', 16))
    result_label.pack()

    # start mainloop
    root.mainloop()


if __name__ == "__main__":
    main()
