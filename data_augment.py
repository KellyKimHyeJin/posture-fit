#

# import cv2
import os
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import transforms
#
# # 데이터 증강 설정
# data_transforms = transforms.Compose([
#     transforms.RandomRotation(degrees=15),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
#     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))
# ])
#
# # 비디오 증강 함수
# def augment_video(video_path, output_dir, num_augmented_versions=5):
#     # 비디오 읽기
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)  # 초당 프레임 수
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # 비디오 파일 이름 가져오기
#     video_filename = os.path.splitext(os.path.basename(video_path))[0]
#
#     # 증강된 비디오 저장 디렉토리 생성
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 증강된 비디오 생성
#     for aug in range(num_augmented_versions):
#         # 증강된 비디오 경로 설정
#         output_path = os.path.join(output_dir, f"{video_filename}_augmented_{aug}.mp4")
#         out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (224, 224))
#
#         # 비디오 프레임 증강
#         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 프레임 위치 초기화
#         for _ in range(frame_count):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             # OpenCV 이미지에서 PIL 이미지로 변환
#             pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#             # 데이터 증강 적용
#             augmented_img = data_transforms(pil_img)
#
#             # 증강된 이미지를 OpenCV 포맷으로 변환
#             augmented_frame = cv2.cvtColor(np.array(augmented_img), cv2.COLOR_RGB2BGR)
#
#             # 증강된 프레임 저장
#             out.write(augmented_frame)
#
#         out.release()  # 비디오 파일 닫기
#     cap.release()
#
# # 데이터 디렉토리에 있는 모든 비디오 파일에 증강 적용
# video_dir = 'data'
# output_dir = 'augmented_videos'
# for video_file in os.listdir(video_dir):
#     if video_file.endswith(".mp4"):
#         augment_video(os.path.join(video_dir, video_file), output_dir)
# import numpy as np
# import os
# from typing import List, Tuple
# import random
#
#
# def augment_video(input_path: str, output_dir: str) -> None:
#     """
#     Perform various augmentations on input video and save results
#     """
#     print(f"\nProcessing video: {input_path}")
#
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Output directory: {output_dir}")
#
#     # Read video
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         print(f"Failed to open video: {input_path}")
#         return
#
#     frames = []
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#         frame_count += 1
#     cap.release()
#
#     print(f"Read {frame_count} frames from video")
#
#     if not frames:
#         print("No frames were read from the video")
#         return
#
#     # Get video properties
#     height = frames[0].shape[0]
#     width = frames[0].shape[1]
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")
#
#     # Get base filename without extension
#     base_name = os.path.splitext(os.path.basename(input_path))[0]
#
#     # Apply different augmentations
#     augmentations = {
#         'horizontal_flip': horizontal_flip,
#         'random_rotation': random_rotation,
#         'random_crop': random_crop,
#         'frame_skip': frame_skip,
#         'time_reverse': time_reverse,
#         'perspective_transform': perspective_transform
#     }
#
#     for aug_name, aug_func in augmentations.items():
#         print(f"\nApplying {aug_name} augmentation...")
#         try:
#             # Apply augmentation
#             augmented_frames = aug_func(frames.copy())
#
#             # Save augmented video
#             output_path = os.path.join(output_dir, f"{base_name}_{aug_name}.mp4")
#             save_video(augmented_frames, output_path, fps)
#         except Exception as e:
#             print(f"Error during {aug_name} augmentation: {str(e)}")
#
#
# def horizontal_flip(frames: List[np.ndarray]) -> List[np.ndarray]:
#     """Flip frames horizontally"""
#     return [cv2.flip(frame, 1) for frame in frames]
#
#
# def random_rotation(frames: List[np.ndarray], max_angle: int = 10) -> List[np.ndarray]:
#     """Rotate frames by a random angle between -max_angle and max_angle degrees"""
#     angle = random.uniform(-max_angle, max_angle)
#     height, width = frames[0].shape[:2]
#     center = (width // 2, height // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#
#     return [cv2.warpAffine(frame, rotation_matrix, (width, height)) for frame in frames]
#
#
# def random_crop(frames: List[np.ndarray], crop_ratio: float = 0.9) -> List[np.ndarray]:
#     """Randomly crop frames while maintaining most of the content"""
#     height, width = frames[0].shape[:2]
#     crop_height = int(height * crop_ratio)
#     crop_width = int(width * crop_ratio)
#
#     # Random crop position
#     start_y = random.randint(0, height - crop_height)
#     start_x = random.randint(0, width - crop_width)
#
#     cropped_frames = [
#         frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
#         for frame in frames
#     ]
#
#     # Resize back to original dimensions
#     return [cv2.resize(frame, (width, height)) for frame in cropped_frames]
#
#
# def frame_skip(frames: List[np.ndarray], skip_rate: int = 2) -> List[np.ndarray]:
#     """Skip frames to simulate speed changes"""
#     return frames[::skip_rate]
#
#
# def time_reverse(frames: List[np.ndarray]) -> List[np.ndarray]:
#     """Reverse the order of frames"""
#     return frames[::-1]
#
#
# def perspective_transform(frames: List[np.ndarray], strength: float = 0.1) -> List[np.ndarray]:
#     """Apply perspective transform to simulate camera position changes"""
#     height, width = frames[0].shape[:2]
#
#     # Define perspective transform points
#     offset = int(width * strength)
#     src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
#     dst_points = np.float32([
#         [offset, offset],
#         [width - offset, offset],
#         [0, height],
#         [width, height]
#     ])
#
#     # Calculate perspective transform matrix
#     matrix = cv2.getPerspectiveTransform(src_points, dst_points)
#
#     return [cv2.warpPerspective(frame, matrix, (width, height)) for frame in frames]
#
#
# def save_video(frames: List[np.ndarray], output_path: str, fps: int) -> None:
#     """Save frames as video"""
#     if not frames:
#         print(f"No frames to save for {output_path}")
#         return
#
#     height, width = frames[0].shape[:2]
#
#     # Try different codecs
#     codecs = ['XVID', 'avc1', 'mp4v']
#     saved = False
#
#     for codec in codecs:
#         try:
#             fourcc = cv2.VideoWriter_fourcc(*codec)
#             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
#             if not out.isOpened():
#                 print(f"Failed to open video writer with codec {codec}")
#                 continue
#
#             for frame in frames:
#                 success = out.write(frame)
#                 if not success:
#                     raise Exception(f"Failed to write frame with codec {codec}")
#
#             out.release()
#
#             # Verify file was created and has size
#             if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
#                 print(f"Successfully saved video with codec {codec}: {output_path}")
#                 print(f"File size: {os.path.getsize(output_path)} bytes")
#                 saved = True
#                 break
#             else:
#                 print(f"File was not created properly with codec {codec}")
#
#         except Exception as e:
#             print(f"Error with codec {codec}: {str(e)}")
#             if out is not None:
#                 out.release()
#             if os.path.exists(output_path):
#                 os.remove(output_path)
#
#     if not saved:
#         print(f"Failed to save video with any codec: {output_path}")
#
#
# if __name__ == "__main__":
#     video_dir = 'data'
#     output_dir = 'augmented_videos'
#
#     print(f"Looking for videos in: {video_dir}")
#
#     if not os.path.exists(video_dir):
#         print(f"Error: Video directory '{video_dir}' does not exist")
#         exit(1)
#
#     videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
#     print(f"Found {len(videos)} videos")
#
#     if not videos:
#         print("No .mp4 videos found in the directory")
#         exit(1)
#
#     for video_file in videos:
#         video_path = os.path.join(video_dir, video_file)
#         augment_video(video_path, output_dir)

# import os
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# import random
#
#
# def create_transforms():
#     """비디오 프레임에 적용할 augmentation 기법들을 정의합니다."""
#     return {
#         'horizontal_flip': transforms.Compose([
#             transforms.RandomHorizontalFlip(p=1.0)
#         ]),
#         'random_rotation': transforms.Compose([
#             transforms.RandomRotation(degrees=10)
#         ]),
#         'random_crop': transforms.Compose([
#             transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0))
#         ]),
#         'perspective': transforms.Compose([
#             transforms.RandomPerspective(distortion_scale=0.2, p=1.0)
#         ])
#     }
#
#
# def augment_video(video_path: str, output_dir: str) -> None:
#     """비디오를 읽고 다양한 augmentation을 적용한 후 저장합니다."""
#     print(f"\nProcessing video: {video_path}")
#
#     # 출력 디렉토리 생성
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Output directory: {output_dir}")
#
#     # 비디오 읽기
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Failed to open video: {video_path}")
#         return
#
#     # 비디오 속성 가져오기
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     print(f"Frame count: {frame_count}, FPS: {fps}")
#
#     # 모든 프레임 읽기
#     frames = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()
#
#     if not frames:
#         print("No frames were read from the video")
#         return
#
#     # 비디오 파일 이름 가져오기
#     base_name = os.path.splitext(os.path.basename(video_path))[0]
#
#     # Augmentation 정의
#     transforms_dict = create_transforms()
#
#     # 각 augmentation 적용
#     for aug_name, transform in transforms_dict.items():
#         print(f"\nApplying {aug_name} augmentation...")
#         try:
#             output_path = os.path.join(output_dir, f"{base_name}_{aug_name}.mp4")
#             augmented_frames = []
#
#             for frame in frames:
#                 # OpenCV BGR을 RGB로 변환
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 pil_image = Image.fromarray(rgb_frame)
#
#                 # Augmentation 적용
#                 augmented_pil = transform(pil_image)
#
#                 # PIL 이미지를 다시 OpenCV 형식으로 변환
#                 augmented_frame = cv2.cvtColor(np.array(augmented_pil), cv2.COLOR_RGB2BGR)
#                 augmented_frames.append(augmented_frame)
#
#             # 시간 기반 augmentation 적용
#             if aug_name == 'frame_skip':
#                 augmented_frames = augmented_frames[::2]  # 프레임 건너뛰기
#             elif aug_name == 'time_reverse':
#                 augmented_frames = augmented_frames[::-1]  # 시간 역전
#
#             # 비디오 저장
#             save_video(augmented_frames, output_path, fps)
#
#         except Exception as e:
#             print(f"Error during {aug_name} augmentation: {str(e)}")
#
#
# def save_video(frames: list, output_path: str, fps: int) -> None:
#     """증강된 프레임들을 비디오로 저장합니다."""
#     if not frames:
#         print(f"No frames to save for {output_path}")
#         return
#
#     height, width = frames[0].shape[:2]
#
#     # 여러 코덱 시도
#     codecs = ['XVID', 'avc1', 'mp4v']
#     saved = False
#
#     for codec in codecs:
#         try:
#             fourcc = cv2.VideoWriter_fourcc(*codec)
#             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
#             if not out.isOpened():
#                 print(f"Failed to open video writer with codec {codec}")
#                 continue
#
#             for frame in frames:
#                 success = out.write(frame)
#                 if not success:
#                     raise Exception(f"Failed to write frame with codec {codec}")
#
#             out.release()
#
#             if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
#                 print(f"Successfully saved video with codec {codec}: {output_path}")
#                 print(f"File size: {os.path.getsize(output_path)} bytes")
#                 saved = True
#                 break
#
#         except Exception as e:
#             print(f"Error with codec {codec}: {str(e)}")
#             if 'out' in locals():
#                 out.release()
#             if os.path.exists(output_path):
#                 os.remove(output_path)
#
#     if not saved:
#         print(f"Failed to save video with any codec: {output_path}")
#
#
# if __name__ == "__main__":
#     video_dir = 'data'
#     output_dir = 'augmented_videos'
#
#     print(f"Looking for videos in: {video_dir}")
#
#     if not os.path.exists(video_dir):
#         print(f"Error: Video directory '{video_dir}' does not exist")
#         exit(1)
#
#     videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
#     print(f"Found {len(videos)} videos")
#
#     if not videos:
#         print("No .mp4 videos found in the directory")
#         exit(1)
#
#     for video_file in videos:
#         video_path = os.path.join(video_dir, video_file)
#         augment_video(video_path, output_dir)

import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import random

# 데이터 증강 설정
data_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # 랜덤 회전
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # 랜덤 자르기
    transforms.RandomHorizontalFlip()  # 좌우 반전
])


# 프레임 건너뛰기 설정
def skip_frames(cap, skip_probability=0.2):
    if random.random() < skip_probability:
        return cap.read()[1]
    return None


# 시점 변환 (카메라 위치 변환 시뮬레이션)
def perspective_transform(frame):
    h, w = frame.shape[:2]
    src_pts = np.float32([
        [0, 0], [w, 0], [0, h], [w, h]
    ])
    dst_pts = np.float32([
        [random.randint(-20, 20), random.randint(-20, 20)],
        [w - random.randint(-20, 20), random.randint(-20, 20)],
        [random.randint(-20, 20), h - random.randint(-20, 20)],
        [w - random.randint(-20, 20), h - random.randint(-20, 20)]
    ])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, matrix, (w, h))


# 비디오 증강 함수
def augment_video(video_path, output_dir, num_augmented_versions=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    for aug in range(num_augmented_versions):
        output_path = os.path.join(output_dir, f"{video_filename}_augmented_{aug}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (224, 224))

        frames = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 건너뛰기
            skipped_frame = skip_frames(cap)
            if skipped_frame is not None:
                frame = skipped_frame

            # 시점 변환
            frame = perspective_transform(frame)

            # PIL 이미지 변환 및 데이터 증강
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            augmented_img = data_transforms(pil_img)
            augmented_frame = cv2.cvtColor(np.array(augmented_img), cv2.COLOR_RGB2BGR)

            frames.append(augmented_frame)

        # 시간 역전 적용
        if random.random() < 0.5:
            frames.reverse()

        for frame in frames:
            out.write(frame)

        out.release()
    cap.release()


# 데이터 디렉토리에 있는 모든 비디오 파일에 증강 적용
video_dir = 'data'
output_dir = 'augmented_videos'
for video_file in os.listdir(video_dir):
    if video_file.endswith(".mp4"):
        augment_video(os.path.join(video_dir, video_file), output_dir)