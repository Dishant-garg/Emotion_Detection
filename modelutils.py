import tensorflow as tf
import tensorflow.keras.layers as L
import os

def load_model():
   
    saved_model_path = os.path.join('Models','model_main2.json')
    with open(saved_model_path , 'r') as json_file:
     json_savedModel = json_file.read()
    model = tf.keras.models.model_from_json(json_savedModel)
    weights_path = os.path.join('Models', 'best_model2.weights.h5')
    model.load_weights(weights_path)
    model.compile(loss='sparse_categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    
    return model


