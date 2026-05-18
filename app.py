from flask import Flask, render_template, Response, jsonify
import cv2, os, joblib
import numpy as np

from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

known_embeddings = joblib.load("models/embeddings.pkl")
known_names = joblib.load("models/names.pkl")
normalizer = joblib.load("models/normalizer.pkl")

face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1)

os.makedirs("attendance", exist_ok=True)
attendance_file = "attendance/attendance.csv"
if not os.path.exists(attendance_file):

    with open(attendance_file, "w") as f:

        f.write("Name,Date,Time\n")

marked_names = set()

face_count = 0

cap = cv2.VideoCapture(0)

def generate_frames():
    global face_count

    while True:
        success, frame = cap.read()

        if not success:
            break

        faces = face_app.get(frame)
        face_count = len(faces)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            embedding = normalizer.transform(
                [face.embedding]
            )

            similarities = cosine_similarity(
                embedding,
                known_embeddings
            )[0]

            idx = np.argmax(similarities)
            similarity = similarities[idx]
            name = known_names[idx]

            if similarity < 0.70:
                name = "Unknown"
                color = (0, 0, 255)

            else:

                color = (0, 255, 0)
                if name not in marked_names:
                    marked_names.add(name)
                    current_date = datetime.now().strftime(
                        "%d-%m-%Y"
                    )

                    current_time = datetime.now().strftime(
                        "%H:%M:%S"
                    )

                    with open(attendance_file, "a") as f:

                        f.write(
                            f"{name},{current_date},{current_time}\n"
                        )

            cv2.rectangle(frame,(x1, y1),(x2, y2),color,3
            )

            cv2.putText(frame,f"{name} ({similarity:.2f})",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2
            )

        ret, buffer = cv2.imencode('.jpg',frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

@app.route('/')

def index():
    attendance = []
    if os.path.exists(attendance_file):

        with open(attendance_file, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                row = line.strip().split(",")
                if len(row) == 3:
                    attendance.append(row)

    return render_template(

        "index.html",
        registered_count=len(known_names),
        face_count=face_count,
        attendance=attendance[::-1]

    )

@app.route('/data')

def data():
    attendance = []
    if os.path.exists(attendance_file):
        with open(attendance_file, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                row = line.strip().split(",")
                if len(row) == 3:

                    attendance.append(row)

    return jsonify({

        "face_count": face_count,
        "attendance": attendance[::-1]

    })

@app.route('/video')

def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'

    )

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )