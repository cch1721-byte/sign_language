import os
import cv2
import numpy as np
from tqdm import tqdm # pip install tqdm 필수

def build_sign_dataset_npy(source_root, save_root):
    # 폴더 리스트업
    categories = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    print(f"총 {len(categories)}개의 카테고리 작업을 시작합니다.")

    # tqdm을 사용하면 실시간 로딩바가 생깁니다.
    for idx, category in enumerate(tqdm(categories, desc="전체 진행도")):
        source_category_path = os.path.join(source_root, category)
        save_category_path = os.path.join(save_root, category)
        
        os.makedirs(save_category_path, exist_ok=True)
        
        video_files = [f for f in os.listdir(source_category_path) if f.lower().endswith(('.mp4', '.avi'))]
        
        for v_file in video_files:
            video_path = os.path.join(source_category_path, v_file)
            save_path = os.path.join(save_category_path, os.path.splitext(v_file)[0] + '.npy')
            
            if os.path.exists(save_path):
                continue
                
            try:
                frames = extract_frames(video_path)
                if frames is not None:
                    np.save(save_path, frames)
            except Exception as e:
                print(f"\n[오류] {v_file}: {e}")
        
        # 50개가 아니라 매 폴더마다 확인하고 싶다면 아래처럼 수정 (tqdm을 안 쓸 경우)
        # print(f" 현재 {idx + 1}/{len(categories)}: {category} 완료")

def extract_frames(video_path):
    # GPU 가속을 지원하는 OpenCV 빌드라면 cv2.cuda를 쓰겠지만, 
    # 일반적인 환경에서는 아래 설정이 읽기 속도 최적화에 도움됩니다.
    cap = cv2.VideoCapture(video_path)
    
    # 영상 읽기 가속을 위한 힌트 (백엔드 설정)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize 작업은 CPU 멀티코어를 사용합니다.
        frame = cv2.resize(frame, (128, 128)) 
        frames.append(frame)
        
    cap.release()
    return np.array(frames, dtype=np.uint8) if frames else None

if __name__ == "__main__":
    SOURCE = r"D:\VITA\NIA_SignLanguage_Dataset\train"
    SAVE = r"D:\VITA\NIA_SignLanguage_Dataset\train_npy"
    build_sign_dataset_npy(SOURCE, SAVE)