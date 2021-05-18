from tensorflow import keras
train = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_set = train.flow_from_directory("image/training/",
                                     target_size=(200,200),
                                     batch_size=32,
                                     class_mode="binary")

validation_set = validation.flow_from_directory("image/validtion/",
                                     target_size=(200,200),
                                     batch_size=32,
                                     class_mode="binary")

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=16, kernel_size=3, input_shape=(200,200,3), activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dense(units=1, activation="sigmoid"))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_set,epochs=10,validation_data=validation_set)

model.save("Cat_Dog_Classification.h5")
