from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA

save_dir = "face_db"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_size = (100, 100)

def load_face_db():
    face_db = {}
    for fname in os.listdir(save_dir):
        if fname.endswith(".npy"):
            name = os.path.splitext(fname)[0]
            path = os.path.join(save_dir, fname)
            face_db[name] = np.load(path)
            print(f"'{name}' 로드 완료 - {face_db[name].shape[0]}개")
    return face_db

def recognize_face():
    face_db = load_face_db()
    all_faces = []
    labels = []
    for name, faces in face_db.items():
        for f in faces:
            all_faces.append(f)
            labels.append(name)

    all_faces = np.array(all_faces)
    if len(all_faces) < 2:
        print("2명 이상의 얼굴이 필요합니다.")
        return

    pca = PCA(n_components=50)
    pca.fit(all_faces)
    face_embeddings = pca.transform(all_faces)

    cap = cv2.VideoCapture(0)
    print("실시간 얼굴 인식을 시작합니다. 'q'를 눌러 종료.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, face_size).flatten().reshape(1, -1)
            input_emb = pca.transform(face_resized)

            sims = cosine_similarity(input_emb, face_embeddings)[0]
            best_idx = np.argmax(sims)
            name = labels[best_idx]
            confidence = sims[best_idx]

            label = f"{name} ({confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


recognize_face()