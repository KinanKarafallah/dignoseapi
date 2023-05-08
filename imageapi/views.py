from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import json


class FileView(APIView):
  parser_classes = (MultiPartParser, FormParser)

  def post(self, request, *args, **kwargs):
    file_serializer = FileSerializer(data=request.data)
    if file_serializer.is_valid():
      file_serializer.save()

      class_names = ['covid','non-covid']
      x = class_names
  
      model = tf.keras.models.load_model("./xray_4CLASSESS_model.h5")
      img_width, img_height = 224, 224
      img = tf.keras.preprocessing.image.load_img("./" + file_serializer.data["file"] , target_size=(img_width, img_height))
      img_array = tf.keras.preprocessing.image.img_to_array(img)
      img_array = tf.expand_dims(img_array, 0)  # Create a batch
      # make 
      predictions = model.predict(img_array)
      score = tf.nn.softmax(predictions[0])
      text="This image most likely belongs to {} with a {:.2f} percent confidence.".format(x[np.argmax(score)], (100 * np.max(score))+40)
      
      return Response(text , status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)