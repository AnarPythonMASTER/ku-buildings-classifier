import os
import json
import numpy as np
import tensorflow as tf

# image predictor function
def predict_single_image(image_path, model, class_mapping):
    print(f"\nAnalyzing image: {image_path}...")
    
    img_size = (model.input_shape[1], model.input_shape[2])
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    
    predicted_index = int(np.argmax(predictions[0]))
    confidence_score = float(np.max(predictions[0]) * 100)
    
    predicted_building = class_mapping[predicted_index]
    
    print("-" * 30)
    print(f"Prediction:  {predicted_building.upper()}")
    print(f"Confidence:  {confidence_score:.2f}%")
    print("-" * 30)
    
    return predicted_building, confidence_score


# execution
test_folder = r"C:\Users\Shahbaz\Desktop\dl\test_images"
MAPPING_PATH = r"C:\Users\Shahbaz\Desktop\dl\models\class_mapping.json"
MODEL_PATH = r"C:\Users\Shahbaz\Desktop\dl\models\cnn_model_increased_image.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# Load the dictionary into memory ONCE before the loop
with open(MAPPING_PATH, 'r') as f:
    loaded_class_mapping = {int(k): v for k, v in json.load(f).items()}


for filename in os.listdir(test_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Skipping unsupported or non-image file: {filename}")
        continue 
        
    full_image_path = os.path.join(test_folder, filename)
    predict_single_image(full_image_path, model, loaded_class_mapping)
    
def salam(x):
    print(x)

salam('Anar')

def saqol(x):
    print(x)

saqol("Shahbaz")

def bashuvaa(x):
    print(x=(x,"Bashuvaa"))

bashuvaa(x="mama qurban")
    