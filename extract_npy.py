import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# 1. 미디어파이프 모델 설정
mp_holistic = mp.solutions.holistic

def extract_landmarks(results):
    """
    각 프레임에서 포즈, 왼손, 오른손 좌표를 추출하여 하나의 배열로 합칩니다.
    """
    # 포즈 (33개 랜드마크 * 4: x, y, z, visibility)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # 왼손 (21개 랜드마크 * 3: x, y, z)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # 오른손 (21개 랜드마크 * 3: x, y, z)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

def process_video(video_path, holistic_model):
    """
    영상을 읽어 프레임별 좌표 배열을 생성합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    sequence = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR을 RGB로 변환하여 MediaPipe 처리
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # 성능 향상을 위해 쓰기 권한 제한
        results = holistic_model.process(image)
        image.flags.writeable = True
        
        # 좌표 추출 후 리스트에 추가
        landmarks = extract_landmarks(results)
        sequence.append(landmarks)
        
    cap.release()
    return np.array(sequence, dtype=np.float32)

def build_sign_dataset_npy(source_root, save_root):
    # 폴더 리스트업 (한글 폴더명도 처리 가능)
    categories = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    print(f"[*] 총 {len(categories)}개의 카테고리 분석을 시작합니다.")

    # Holistic 모델 초기화
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for category in tqdm(categories, desc="전체 진행도"):
            source_category_path = os.path.join(source_root, category)
            save_category_path = os.path.join(save_root, category)
            os.makedirs(save_category_path, exist_ok=True)
            
            video_files = [f for f in os.listdir(source_category_path) if f.lower().endswith(('.mp4', '.avi'))]
            
            for v_file in video_files:
                video_path = os.path.join(source_category_path, v_file)
                # 확장자만 .npy로 변경
                save_path = os.path.join(save_category_path, os.path.splitext(v_file)[0] + '.npy')
                
                # 이미 존재하면 건너뛰기 (이어하기 기능)
                if os.path.exists(save_path):
                    continue
                    
                try:
                    npy_data = process_video(video_path, holistic)
                    if npy_data is not None:
                        np.save(save_path, npy_data)
                except Exception as e:
                    print(f"\n[오류] {v_file} 처리 실패: {e}")

if __name__ == "__main__":
    # 서버의 실제 경로에 맞춰 수정하세요.
    SOURCE = "./train"      # 수어 영상이 단어별 폴더로 정리된 곳
    SAVE = "./train_npy"   # 결과 npy 파일이 저장될 곳
    
    build_sign_dataset_npy(SOURCE, SAVE)
    print("\n[최종] 모든 영상의 좌표 추출이 완료되었습니다.")