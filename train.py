import os, cv2, random, joblib
import numpy as np

from insightface.app import FaceAnalysis
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)

DATASET = "Dataset"
MODEL_DIR = "models"

THRESHOLD = 0.70

random.seed(42)
np.random.seed(42)

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)

def augment(img):

    return [
        img,
        cv2.flip(img, 1)
    ]

train = []
val = []
test = []

for person in os.listdir(DATASET):

    path = os.path.join(DATASET, person)
    if not os.path.isdir(path):
        continue

    images = [
        (os.path.join(path, img), person)
        for img in os.listdir(path)
    ]

    random.shuffle(images)
    n = len(images)
    train += images[:int(0.6*n)]
    val += images[
        int(0.6*n):int(0.8*n)
    ]

    test += images[int(0.8*n):]

print(f"\nTrain:{len(train)}")
print(f"Validation:{len(val)}")
print(f"Test:{len(test)}")

person_embeddings = {}

print("\nExtracting embeddings...")
for img_path, person in train:
    img = cv2.imread(img_path)
    if img is None:
        continue

    for aug in augment(img):

        try:
            faces = app.get(aug)
            if len(faces) == 0:
                continue

            face = max(faces,key=lambda x:(x.bbox[2]-x.bbox[0]) *(x.bbox[3]-x.bbox[1]))

            embedding = face.embedding
            if person not in person_embeddings:\
                person_embeddings[person] = []
            person_embeddings[person].append(
                embedding
            )
        except Exception as e:
            print(e)

known_embeddings = []
known_names = []

for person, embeds in person_embeddings.items():
    mean_embedding = np.mean(embeds,axis=0)

    known_embeddings.append(mean_embedding)
    known_names.append(person)

normalizer = Normalizer(norm='l2')
known_embeddings = normalizer.fit_transform(np.array(known_embeddings))

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(known_embeddings,f"{MODEL_DIR}/embeddings.pkl")
joblib.dump(known_names,f"{MODEL_DIR}/names.pkl")
joblib.dump(normalizer,f"{MODEL_DIR}/normalizer.pkl")

print("\nModels Saved Successfully")

def evaluate(data, name):

    y_true = []
    y_pred = []
    for img_path, actual in data:
        img = cv2.imread(img_path)
        if img is None:
            continue
        try:
            faces = app.get(img)
            if len(faces) == 0:
                continue

            face = max(faces,key=lambda x:(x.bbox[2]-x.bbox[0]) *(x.bbox[3]-x.bbox[1]))
            embedding = normalizer.transform([face.embedding])
            similarities = cosine_similarity(embedding,known_embeddings)[0]

            best_index = np.argmax(similarities)
            best_similarity = similarities[best_index]
            prediction = (
                known_names[best_index]
                if best_similarity > THRESHOLD
                else "Unknown"
            )

            y_true.append(actual)
            y_pred.append(prediction)

        except Exception as e:
            print(e)

    if len(y_true) == 0:
        print("\nNo Predictions Made")
        return

    print(f"\n{name} Results")
    print("-" * 35)

    print(f"Accuracy : {accuracy_score(y_true,y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_true,y_pred,average='weighted',zero_division=0)*100:.2f}%")
    print(f"Recall   : {recall_score(y_true,y_pred,average='weighted',zero_division=0)*100:.2f}%")
    print(f"F1 Score : {f1_score(y_true,y_pred,average='weighted',zero_division=0)*100:.2f}%")
    print("\nConfusion Matrix")
    print("-" * 35)

    cm = confusion_matrix(y_true,y_pred)
    print(cm)

evaluate(val,"Validation")
evaluate(test,"Test")
print("\nTraining Completed Successfully")