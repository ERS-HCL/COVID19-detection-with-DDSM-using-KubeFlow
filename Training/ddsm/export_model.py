import os
import pickle
from keras.optimizers import Adam
import keras
from keras.layers import Input,Dropout
from keras.layers.core import Dense
from keras.models import Model


#sudo python3 train_covid_args.py -i /home/amd01/ddsm/dataset/images/images -o ./output -ow best_weights.h5 -e 2 -b 32 -l 0.001 -c ./ -iw ./best_weights.h5




def main():
   # parser config
    # config_file = "./sample_config.ini"
    # cp = ConfigParser()
    # cp.read(config_file)
   
    image_dimension=224
        
    base_model_class = keras.applications.densenet.DenseNet121
    input_shape=(image_dimension, image_dimension, 1)    
    img_input = Input(shape=input_shape)
    base_model = base_model_class(
        include_top=False,
        input_tensor=img_input,
        input_shape=input_shape,
        weights=None,
        )
    x = base_model.output
    x=keras.layers.GlobalAveragePooling2D()(x)
    
    x = Dense(512,activation='relu',name="dense_512")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(15, activation="sigmoid", name="my_prediction")(x)
    model = Model(inputs=img_input, outputs=predictions)
    print("trying to load the weight")
    model.load_weights("./models/best_weights.h5")
    print("loaded the weight")
    
    # if show_model_summary:
    #     print(model.summary())
    # model.save_weights('./my_model_weights.h5')

    model.save_weights('gs://ddsmplus/my_model_weights.h5')

if __name__ == "__main__":
    main()