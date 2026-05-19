A Real-Time attendance system using face recognition that marks attendance automatically using your webcam.
It captures faces via webcam,compares them with stored embeddings, and logs Name, Date, Time into a CSV file.

Tech Used & Purpose

Flask → Web server + dashboard 
OpenCV → Webcam capture, drawing boxes/text on faces
InsightFace (buffalo_l) → Face detection + embeddings 
Scikit-learn (cosine similarity, normalizer) → Compare embeddings, normalize vectors
Joblib → Save/load trained embeddings, names, and normalizer
HTML → Dashboard UI showing live video + attendance records

