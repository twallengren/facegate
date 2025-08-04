from fastapi import FastAPI, UploadFile, File
from typing import List
import face_recognition
import numpy as np
import io

app = FastAPI()

def get_face_encoding(image_bytes: bytes):
    try:
        img = face_recognition.load_image_file(io.BytesIO(image_bytes))
        encodings = face_recognition.face_encodings(img)
        return encodings[0] if encodings else None
    except Exception:
        return None

def compute_pairwise_distances(encodings: List[np.ndarray]) -> np.ndarray:
    n = len(encodings)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(encodings[i] - encodings[j])
            distances[i][j] = d
            distances[j][i] = d
    return distances

@app.post("/validate")
async def validate_images(files: List[UploadFile] = File(...)):
    encodings = []
    valid_indices = []
    invalid_files = []
    reasons = {}

    file_bytes = [await file.read() for file in files]
    encoding_map = []

    for i, data in enumerate(file_bytes):
        encoding = get_face_encoding(data)
        if encoding is None:
            invalid_files.append(files[i].filename)
            reasons[files[i].filename] = "no_face"
        else:
            encodings.append(encoding)
            encoding_map.append((i, files[i].filename))

    # Early exit: less than 2 valid faces
    if len(encodings) < 2:
        return {
            "valid": [],
            "invalid": [f.filename for f in files],
            "reason": {f.filename: "too_few_valid_faces" for f in files}
        }

    distances = compute_pairwise_distances(encodings)
    threshold = 0.6  # typical face_recognition threshold

    for idx, (original_index, filename) in enumerate(encoding_map):
        avg_distance = np.mean([
            distances[idx][j] for j in range(len(encodings)) if j != idx
        ])
        if avg_distance > threshold:
            invalid_files.append(filename)
            reasons[filename] = f"face_inconsistent (avg_distance={avg_distance:.3f})"
        else:
            valid_indices.append(original_index)

    valid_files = [files[i].filename for i in valid_indices]

    return {
        "valid": valid_files,
        "invalid": invalid_files,
        "reason": reasons
    }
