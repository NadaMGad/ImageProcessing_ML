import cv2
import numpy as np
from keras._tf_keras.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_uploaded_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to VGG16 input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # Preprocess for VGG16
    features = model.predict(img)
    return features.flatten()

def find_similar_places(city_name, user_features, data, threshold=0.8, top_n=3):
    filtered_data = data[data['CityName'] == city_name]
    if filtered_data.empty:
        return []

    X = filtered_data.iloc[:, 2:].values  # Extract feature columns
    similarities = cosine_similarity([user_features], X)
    similar_indices = np.argsort(similarities.flatten())[::-1]

    places = []
    seen_places = set()

    for idx in similar_indices:
        if len(places) >= top_n:
            break
        similarity_score = similarities[0, idx]
        if similarity_score >= threshold:
            place_name = filtered_data.iloc[idx]['PlaceName']
            if place_name not in seen_places:
                print(f"Place: {place_name}, Similarity Score: {similarity_score}")
                places.append(place_name)
                seen_places.add(place_name)

    return places

def find_similar_item(place_name, user_features, data, threshold=0.8):
    filtered_data = data[data['PlaceName'] == place_name]
    if filtered_data.empty:
        return None

    X = filtered_data.iloc[:, 2:].values  # Extract feature columns
    similarities = cosine_similarity([user_features], X)
    similar_indices = np.argsort(similarities.flatten())[::-1]

    for idx in similar_indices:
        similarity_score = similarities[0, idx]
        if similarity_score >= threshold:
            item_name = filtered_data.iloc[idx]['ItemName']
            return item_name

    return None
