from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers.core import Dense, Dropout, Flatten

def get_model(input_shape, config, top = True):
    input_img = Input(input_shape)

    def __body(input_img):
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_img)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        embedding = Dense(128, activation='relu')(x)
        return embedding

    def __head(embedding):
        x   = Dropout(0.25)(embedding)
        out = Dense(config["data"]["num_classes"], activation='softmax')(x)
        return out

    x = __body(input_img)
    if top: x = __head(x)

    model = Model(inputs=input_img, outputs=x)
    return model