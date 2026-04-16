import pandas as pd
import os
import shutil
import re
from tqdm import tqdm

def organize_dataset(csv_path, source_video_dir, target_root_dir):
    if not os.path.exists(csv_path):
        print(f"[오류] CSV 파일을 찾을 수 없습니다: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path, encoding='cp949')
    except:
        df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 1. 파일 시스템 스캔 및 'ID 매핑' 생성
    print(f"[*] 스캔 시작: {source_video_dir}")
    # key: 파일명 앞부분(ID), value: [전체 경로 리스트]
    id_to_paths = {}
    
    # 수정 제안: 모든 하위 폴더의 mp4를 다 찾아내는 방식
    video_files = {}
    for root, dirs, files in os.walk("./video"):
        for f in files:
            if f.endswith('.mp4'):
                # 파일명에서 ID 추출 (파일명과 CSV ID가 일치해야 함)
                file_id = os.path.splitext(f)[0] 
                video_files[file_id] = os.path.join(root, f)

    print(f"[*] 스캔 완료: 실제 영상 파일 {len(video_files)}개 발견")

    # 2. 정리 작업
    success_count = 0
    missing_count = 0
    
    desc_name = os.path.basename(csv_path)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"정리 중 ({desc_name})"):
        csv_filename = str(row['Filename']).strip()
        # CSV 파일명에서도 ID 추출 (예: ..._F.mp4 -> ..._F 제외)
        csv_file_no_ext = os.path.splitext(csv_filename)[0]
        csv_id = "_".join(csv_file_no_ext.split('_')[:-1]) if '_' in csv_file_no_ext else csv_file_no_ext
        
        # 라벨 정제
        raw_label = str(row['Kor'])
        clean_label = re.sub(r'[\/:*?"<>|]', '_', raw_label.replace('\n', ' ').replace('\r', ' ')).strip()
        
        target_dir = os.path.join(target_root_dir, clean_label)
        
        # 해당 ID를 가진 파일이 스캔 결과에 있는지 확인
        if csv_id in id_to_paths:
            os.makedirs(target_dir, exist_ok=True)
            for src_path in id_to_paths[csv_id]:
                file_name = os.path.basename(src_path)
                dst_path = os.path.join(target_dir, file_name)
                
                try:
                    shutil.move(src_path, dst_path)
                    success_count += 1
                except Exception as e:
                    print(f"\n[!] 에러 ({file_name}): {e}")
            # 이동 완료된 ID는 삭제
            del id_to_paths[csv_id]
        else:
            missing_count += 1

    print(f"\n[결과] {desc_name} 정리 완료")
    print(f"- 이동된 총 파일 수: {success_count}개")
    print(f"- 매칭 실패한 ID 수: {missing_count}개")

# --- 설정 구간 ---
SOURCE_VIDEO_ROOT = './video' 
OUTPUT_ROOT = '/NIA_SignLanguage_Dataset'

tasks = [
    {'csv': 'NIA_SEN_train.csv', 'subdir': 'train'},
    {'csv': 'NIA_SEN_val.csv',   'subdir': 'val'}
]

if __name__ == "__main__":
    for task in tasks:
        target_path = os.path.join(OUTPUT_ROOT, task['subdir'])
        organize_dataset(task['csv'], SOURCE_VIDEO_ROOT, target_path)
    print("\n[최종] 모든 각도의 영상이 정리되었습니다.")