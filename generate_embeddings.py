import face_recognition
import os
import pickle
from PIL import Image
import numpy as np

dataset_path = 'dataset'
embeddings_file = 'embeddings.pkl'

# Initialize
known_encodings = []
known_names = []

# 1. Load existing embeddings
if os.path.exists(embeddings_file):
    print(f"[INFO] Found existing {embeddings_file}, loading it...")
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
        known_encodings = data.get("encodings", [])
        known_names = data.get("names", [])
    print(f"[INFO] Loaded {len(known_encodings)} existing embeddings.")

existing_names_set = set(known_names)  # For faster lookup

# 2. Process only NEW persons (not already in embeddings)
for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder):
        continue

    if person in existing_names_set:
        print(f"[SKIP] Already processed: {person}")
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        try:
            pil_image = Image.open(img_path).convert("RGB")
            image = np.array(pil_image, dtype=np.uint8)
            image = np.ascontiguousarray(image)

            print(f"[INFO] Processing: {img_path}")
            face_encs = face_recognition.face_encodings(image)

            if face_encs:
                known_encodings.append(face_encs[0])
                known_names.append(person)
            else:
                print(f"[WARNING] No face found in {img_path}")

        except Exception as e:
            print(f"[ERROR] Skipping {img_path} due to: {e}")

# 3. Save updated embeddings
data = {"encodings": known_encodings, "names": known_names}
with open(embeddings_file, "wb") as f:
    pickle.dump(data, f)

print(f"[✅] Saved total {len(known_encodings)} face embeddings to {embeddings_file}")
