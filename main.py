# This is a sample Python script.
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tkinter
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.filedialog as fd
import array as arr
from array import *
import time
import cv2



from tkinter import *
# loading Python Imaging Library
from PIL import ImageTk, Image
# To get the dialog box to open when required
from tkinter import filedialog


def open_img():
    # Select the Imagename from a folder

    x = openfilename()
    if x:
        # opens the image
        img = Image.open(x)
        if (img.size[0] > img.size[1]):
            scale_percent = 150 / img.size[1]  # percent of original size
        else:
            scale_percent = 150 / img.size[0]  # percent of original size

        width = int(img.size[0] * scale_percent)
        height = int(img.size[1] * scale_percent)

        # resize the image and apply a high-quality down sampling filter
        img = img.resize((width, height), Image.ANTIALIAS)

        # PhotoImage class is used to add image to widgets, icons etc
        img = ImageTk.PhotoImage(img)

        # create a label
        panel = Label(root, image = img)

        # set the image as img
        panel.image = img
        panel.grid(row = 2)

filename=""
prediction=6
strr=""

classname=["Стирать при температуре до 30 градусов","Химчистка не разрешена","Нельзя выжимать и сушить в машине","Чистка с использованием углеводорода хлорного этилена монофлотрихлорметана разрешена","Нельзя отбеливать","Гладить при температуре до 110 градусов","Пусто"]
def openfilename():
    global filename
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title ='"pen')
    return filename

def predict():
    global prediction
    global classname
    global pred
    global strr
    img = cv2.imread(filename)
    pred.set("")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=3)
    canny = cv2.Canny(img_erode,10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)


    # Get contours
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()
    cv2.imshow("",output)


    letters = []
    for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
            # hierarchy[i][0]: the index of the next contour of the same level
            # hierarchy[i][1]: the index of the previous contour of the same level
            # hierarchy[i][2]: the index of the first child
            # hierarchy[i][3]: the index of the parent
            # if hierarchy[0][idx][3] == -1:
            area = int(w * h)
            if area > 9000:
                cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
                letter_crop = img[y:y + h, x:x + w]
                # print(letter_crop.shape)

                # Resize letter canvas to square
                size_max = max(w, h)
                letter_square = 255 * np.ones(shape=[size_max, size_max,3], dtype=np.uint8)
                if w > h:
                    # Enlarge image top-bottom
                    # ------
                    # ======
                    # ------
                    y_pos = size_max // 2 - h // 2
                    letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                elif w < h:
                    # Enlarge image left-right
                    # --||--
                    x_pos = size_max // 2 - w // 2
                    letter_square[0:h, x_pos:x_pos + w] = letter_crop
                else:
                    letter_square = letter_crop

                letters.append((x, w, cv2.resize(letter_square, (100, 100), interpolation=cv2.INTER_AREA)))

                # Sort array in place by X-coordinate
    
    cv2.imshow("", output)
    '''
    for i, let in enumerate(letters):
        if i != 0:
            cv2.imshow(str(i), let[2])


    #scale_percent = 30  # percent of original size
    #width = int(img.shape[1] * scale_percent / 100)
    #height = int(img.shape[0] * scale_percent / 100)
    #dim = (width, height)
    #img = cv2.resize(img, dim)
    #cv2.imshow("Input", img)
    # cv2.imshow("gray", gray)
    # cv2.imshow("thresh", thresh)
    #img = cv2.resize(img_erode, dim)
    #cv2.imshow("Enlarged", img_erode)
    #img = cv2.resize(output, dim)
    #cv2.imshow("Output", output)
    #cv2.waitKey(0)
    '''
    class_pred=[]

    model = load_model("best_model.h5")
    for i, let in enumerate(letters):
        flag = 0
        if i != 0:
            img=let[2]
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            prediction = model.predict(x)
            prediction = np.argmax(prediction)
            for c in class_pred:
                if prediction==c:
                    flag=1
            if flag==0:
                class_pred.append(prediction)


    for k in range(len(class_pred)):
        strr=strr+classname[class_pred[k]]+"\n"

    # вывод предсказывания на форму
    pred.set(strr)
    #img = image.load_img(filename, target_size=(100, 100))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #prediction = model.predict(x)
    #prediction = np.argmax(prediction)
    #вывод предсказывания на форму
    #pred.set(classname[prediction])






# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Create a window
    root = Tk()
    root.geometry("600x350")
    pred = StringVar()
    # Set Title as Image Loader
    root.title("Image Loader")

    # Allow Window to be resizable
    root.resizable(width=True, height=True)

    # Create a button and place it into the window using grid layout
    btn = Button(root, text='open image', command=open_img).grid(row = 1, columnspan = 4)

    btn1 = Button(root, text='Predict', command=predict).grid(row=3, columnspan=4)

    label= Label(root, textvariable=pred).grid(row=4, columnspan=2)



    root.mainloop()





    #img = ImageTk.PhotoImage(Image.open(openfile()))

    # Add image to the Canvas Items
    #canvas.create_image(10, 10, anchor=NW, image=img)
   # root.mainloop()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
#model = load_model("D:/распознавание образов/sign_model(6 классов,55 процентов).h5")
#print(model.summary())


#Set the geometry of tkinter frame
