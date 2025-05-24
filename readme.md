# Face PCA 프로젝트

이 프로젝트는 웹캠을 통해 얼굴 데이터를 수집하고, PCA를 통해 Eigenface(고유 얼굴)를 계산하며, 원본 얼굴 이미지를 재구성하는 과정을 보여줍니다.

## 개요

* **opencv.py**: 웹캠에서 얼굴을 감지·등록하여 1차원 벡터로 변환한 후 `face_db/` 폴더에 저장합니다.
* **faces\_pca.ipynb**: 저장된 얼굴 벡터(.npy)를 불러와 PCA를 수행, Eigenface를 추출하고 원본 및 재구성 이미지를 시각화합니다.

## 요구 사항

* Python 3.7 이상
* OpenCV 호환 웹캠
* Jupyter Notebook (faces\_pca.ipynb 실행용)

## 설치 방법

1. 리포지터리 클론:

   ```bash
   git clone https://github.com/yourusername/face-pca-project.git
   cd face-pca-project
   ```
2. (선택) 가상환경 생성 및 활성화:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. 의존성 설치:

   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

## 사용법

### 1. 얼굴 데이터 등록

```bash
python opencv.py --name 사용자이름
```

* 웹캠 창이 뜨며 얼굴이 감지될 때마다 벡터로 저장됩니다.
* `q` 키를 누르면 등록을 종료합니다.
* 벡터 파일은 `face_db/사용자이름.npy`에 저장됩니다.

### 2. PCA 수행 및 시각화

```bash
jupyter notebook faces_pca.ipynb
```

* `face_db/` 폴더의 `.npy` 파일을 읽어옵니다.
* PCA 모델로부터 Eigenface를 추출합니다.
* 원본 이미지, Eigenface 컴포넌트, 재구성 이미지를 표시합니다.
* 노트북 내 `n_components` 값을 조정해 구성 요소 개수를 바꿀 수 있습니다.


4. 브라우저에서 `http://localhost:8888` 접속

## 디렉터리 구조

```
face-pca-project/
├── opencv.py            # 얼굴 벡터 등록 스크립트
├── faces_pca.ipynb      # PCA 분석 및 시각화 노트북
├── face_db/             # 얼굴 벡터(.npy) 저장 폴더
├── requirements.txt     # Python 의존성 목록
└── Dockerfile           # Docker 컨테이너 정의 (선택)
```

## 라이선스

MIT 라이선스 하에 공개되어 있습니다. 자유롭게 사용 및 수정하세요.

---


