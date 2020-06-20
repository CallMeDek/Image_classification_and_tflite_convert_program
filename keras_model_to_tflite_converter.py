from tensorflow.compat.v1.lite import TFLiteConverter

converter = TFLiteConverter.from_keras_model_file("./result/model({0})_weights_imagenet.h5".format(name))
tflite_model = converter.convert()
open("trained_model.tflite", "wb").write(tflite_model)