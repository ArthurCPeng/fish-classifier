import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, metrics, optimizers, losses
import os
import random


extract = False
transfer = False
rebuild = True

#-----------------------------------EXTRACTION OF DATA INTO NUMPY ARRAY-----------------------------------#
classes_all = ["Black Sea Sprat",
               "Gilt-Head Bream",
               "Horse Mackerel",
               "Red Mullet",
               "Red Sea Bream",
               "Sea Bass",
               "Striped Red Mullet",
               "Trout"]
total_classes = len(classes_all)

if extract:
    ###-----READ AND EXTRACT DATA FROM CSV-----###
    filepaths_all = []
    data_size = 0

    for fish_class in classes_all:
        filepaths_1 = os.listdir("fishes/Fish_Dataset/Fish_Dataset/{x}/{x}".format(x=fish_class))
        filepaths_1 = ["fishes/Fish_Dataset/Fish_Dataset/{x}/{x}/".format(x=fish_class)+path for path in filepaths_1]
        #print(len(filepaths_1))
        filepaths_2 = os.listdir("fishes/NA_Fish_Dataset/{}".format(fish_class))
        filepaths_2 = ["fishes/NA_Fish_Dataset/{}/".format(fish_class) + path for path in filepaths_2]
        #print(len(filepaths_2))

        for filepath in filepaths_1 + filepaths_2:
            data_size += 1
            filepaths_all.append([filepath, fish_class])

    print(data_size)

    #Remember to shuffle before splitting the data, otherwise val_accuracy will be very low
    random.shuffle(filepaths_all)


    ###-----SORE THE LABELS AND FILE PATHS IN LISTS-----###
    all_filelist = []
    all_label_list = []

    for filepath in filepaths_all:

        index = filepaths_all.index(filepath)
        freq = 1000
        progress = index / len(filepaths_all) * 100

        if index % freq == freq - 1:
            print("Progress: {}%".format(progress))

        # img = cv2.imread(filepath[0])
        # img = img.reshape(1, 224, 224, 3)

        label = filepath[1]
        label_index = classes_all.index(label)
        label_array = np.zeros([total_classes])
        label_array[label_index] = 1


        all_label_list.append(label_array)
        all_filelist.append(filepath[0])


    all_data = np.array([np.array(Image.open(fname).resize((196,148))) for fname in all_filelist])
    all_label = np.stack(all_label_list)

    val_count = 1000

    train_data = all_data[val_count:,:,:,:]
    train_label = all_label[val_count:,:]

    val_data = all_data[:val_count,:,:,:]
    val_label = all_label[:val_count,:]

    print(all_data.shape)
    print(all_label.shape)

    print(train_data.shape)
    print(val_data.shape)

    print(train_label.shape)
    print(val_label.shape)

    ###-----SAVE ARRAYS IN NPZ FILE-----###
    np.savez("array_data",
             train_data=train_data,
             train_label=train_label,
             val_data=val_data,
             val_label=val_label,
             )



#-----------------------------------LOADING DATA INTO NUMPY ARRAY-----------------------------------#

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
file = open("array_data.npz", "rb")
data = np.load(file)

train_data = data.f.train_data
val_data = data.f.val_data
train_label = data.f.train_label
val_label = data.f.val_label

file.close()

# restore np.load for future normal usage
np.load = np_load_old


if rebuild:
    print("Building model")

    if transfer:
        #-----------------------------------BUILDING A MODEL WITH TRANSFER LEARNING-----------------------------------#
        base_model = tf.keras.applications.MobileNetV2(input_shape=(148, 196, 3),
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False

        global_average_layer = layers.GlobalAveragePooling2D()

        prediction_layer = layers.Dense(8, activation="softmax")

        model = tf.keras.models.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ]
        )
    else:
        #-----------------------------------BUILDING AND TRAINING THE MODEL-----------------------------------#

        model = tf.keras.models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(148, 196, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation="relu"))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(8, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy", #error raised if loss is sparse_categorical_crossentropy
        metrics = ["accuracy"]
    )

else:
    print("Loading existing model")
    model = tf.keras.models.load_model("fish_classifier_original.h5")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy", #error raised if loss is sparse_categorical_crossentropy
        metrics = ["accuracy"]
    )

print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    train_data, train_label,
    validation_data=(val_data, val_label),
    batch_size = 32,
    epochs= 5,
    callbacks= [callback],
    shuffle=True
    # verbose = 2
)

model.save("fish_classifier_original.h5")
