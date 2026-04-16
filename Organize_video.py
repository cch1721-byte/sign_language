import pandas as pd
import os
import shutil
import re
from tqdm import tqdm

def organize_dataset(csv_paths, source_video_dir, target_root_dir):
    # 1. 파일 시스템 스캔: 모든 하위 폴더의 mp4를 찾아서 '순수 ID'별로 그룹화
    print(f"[*] 스캔 시작: {source_video_dir}")
    # key: 순수 ID (각도 제외), value: [각도가 포함된 전체 경로들]
    id_to_paths = {}
    
    # 방향 접미사 리스트
    directions = ('_F', '_R', '_L', '_U', '_D')

    for root, dirs, files in os.walk(source_video_dir):
        for f in tqdm(files, desc="파일 리스트 생성 중", leave=False):
            if f.lower().endswith('.mp4'):
                full_path = os.path.join(root, f)
                file_no_ext = os.path.splitext(f)[0]
                
                # 파일명 끝이 방향 접미사면 제거하여 '순수 ID' 생성
                # 예: NIA_SL_SEN0001_REAL01_F -> NIA_SL_SEN0001_REAL01
                pure_id = file_no_ext
                if file_no_ext.endswith(directions):
                    pure_id = file_no_ext[:-2]
                
                if pure_id not in id_to_paths:
                    id_to_paths[pure_id] = []
                id_to_paths[pure_id].append(full_path)

    print(f"[*] 스캔 완료: 고유 ID {len(id_to_paths)}개 발견")

    # 2. 정리 작업
    success_count = 0
    missing_count = 0
    
    # 여러 CSV 데이터를 하나로 통합 처리
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"[경고] CSV 파일을 찾을 수 없습니다: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path, encoding='cp949')
        except:
            df = pd.read_csv(csv_path, encoding='utf-8')
        
        desc_name = os.path.basename(csv_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"정리 중 ({desc_name})"):
            # CSV의 Filename에서 순수 ID 추출
            csv_filename = str(row['Filename']).strip()
            csv_file_no_ext = os.path.splitext(csv_filename)[0]
            
            # CSV 파일명도 각도 정보가 붙어있을 수 있으므로 제거
            csv_id = csv_file_no_ext
            if csv_file_no_ext.endswith(directions):
                csv_id = csv_file_no_ext[:-2]
            
            # 수어 단어(라벨) 정제 및 폴더 경로 설정
            raw_label = str(row['Kor'])
            clean_label = re.sub(r'[\/:*?"<>|]', '_', raw_label.replace('\n', ' ').replace('\r', ' ')).strip()
            target_dir = os.path.join(target_root_dir, clean_label)
            
            # 매칭 확인 및 이동
            if csv_id in id_to_paths:
                os.makedirs(target_dir, exist_ok=True)
                # 해당 ID를 가진 모든 각도의 파일을 이동
                for src_path in id_to_paths[csv_id]:
                    file_name = os.path.basename(src_path)
                    dst_path = os.path.join(target_dir, file_name)
                    
                    try:
                        shutil.move(src_path, dst_path)
                        success_count += 1
                    except Exception as e:
                        print(f"\n[!] 에러 ({file_name}): {e}")
                
                # 중복 처리 방지를 위해 매칭된 ID는 딕셔너리에서 제거하거나 비움
                del id_to_paths[csv_id]
            else:
                missing_count += 1

    print(f"\n[결과] 모든 작업 완료")
    print(f"- 이동된 총 파일 수: {success_count}개")
    print(f"- 매칭 실패한 ID 수: {missing_count}개")

# --- 설정 구간 ---
SOURCE_VIDEO_ROOT = './video' 
OUTPUT_ROOT = './train'  # 모든 데이터를 train 폴더 하나에 모음

# 합칠 CSV 파일들
csv_files = ['NIA_SEN_train.csv', 'NIA_SEN_val.csv']

if __name__ == "__main__":
    organize_dataset(csv_files, SOURCE_VIDEO_ROOT, OUTPUT_ROOT)
    print(f"\n[최종] 모든 영상이 {OUTPUT_ROOT} 폴더로 정리되었습니다.")