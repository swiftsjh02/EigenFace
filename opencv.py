import os
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA

face_size = (100, 100)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
save_dir = "face_db"
os.makedirs(save_dir, exist_ok=True)  # 저장 폴더 없으면 생성

def register_person(name):
    cap = cv2.VideoCapture(0)
    print(f"'{name}' 등록을 시작합니다. 'q'를 눌러 종료하세요.")
    face_vectors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, face_size)
            face_flattened = face_resized.flatten()
            face_vectors.append(face_flattened)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Register", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 저장
    if face_vectors:
        face_array = np.array(face_vectors)
        save_path = os.path.join(save_dir, f"{name}.npy")
        np.save(save_path, face_array)
        print(f"[{name}] 등록 완료 - {face_array.shape[0]}개의 얼굴 벡터 저장됨 → {save_path}")

register_person("me")