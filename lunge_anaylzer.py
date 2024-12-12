import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
import torch.nn as nn
import torch.nn.functional as F

# class LungePoseAnalyzer:
#     def __init__(self):
#         self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         self.mp_pose = mp.solutions.pose
#
#     def extract_features(self, video_path):
#         video = cv2.VideoCapture(video_path)
#         frame_count = 0
#         angles = {
#             'right_knee': [],
#             'left_knee': [],
#             'right_hip': [],
#             'left_hip': []
#         }
#
#         while video.isOpened():
#             ret, frame = video.read()
#             if not ret:
#                 break
#             frame_count += 1
#
#             # MediaPipe Pose 추출
#             results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#             if results.pose_landmarks:
#                 # 양쪽 무릎과 엉덩이 각도 계산
#                 frame_angles = self.calculate_all_angles(results.pose_landmarks)
#                 for key in angles.keys():
#                     angles[key].append(frame_angles[key])
#
#         video.release()
#
#         # 각 관절별로 가장 낮은 자세(각도가 가장 작을 때)를 찾음
#         lowest_position_angles = {
#             'right_knee_angle': min(angles['right_knee']) if angles['right_knee'] else 180,
#             'left_knee_angle': min(angles['left_knee']) if angles['left_knee'] else 180,
#             'right_hip_angle': min(angles['right_hip']) if angles['right_hip'] else 180,
#             'left_hip_angle': min(angles['left_hip']) if angles['left_hip'] else 180
#         }
#
#         return lowest_position_angles
#
#     def calculate_all_angles(self, landmarks):
#         # 오른쪽 무릎 각도
#         right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
#         right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
#         right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
#
#         # 왼쪽 무릎 각도
#         left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
#         left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
#         left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
#
#         # 오른쪽 엉덩이 각도
#         right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
#
#         # 왼쪽 엉덩이 각도
#         left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
#
#         angles = {
#             'right_knee': self.calculate_angle(right_hip, right_knee, right_ankle),
#             'left_knee': self.calculate_angle(left_hip, left_knee, left_ankle),
#             'right_hip': self.calculate_angle(right_shoulder, right_hip, right_knee),
#             'left_hip': self.calculate_angle(left_shoulder, left_hip, left_knee)
#         }
#
#         return angles
#
#     def calculate_angle(self, point1, point2, point3):
#         vector1 = np.array([point1.x - point2.x, point1.y - point2.y])
#         vector2 = np.array([point3.x - point2.x, point3.y - point2.y])
#
#         dot_product = np.dot(vector1, vector2)
#         magnitude1 = np.linalg.norm(vector1)
#         magnitude2 = np.linalg.norm(vector2)
#
#         cos_theta = dot_product / (magnitude1 * magnitude2)
#         angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
#
#         return np.degrees(angle)
#
#     def process_lunge_video(self, video_path):
#         # 비디오에서 특징을 추출하고 반환하는 메서드
#         features = self.extract_features(video_path)
#         return [features]
#
#
# def extract_features_from_video(video_paths, labels):
#     data = []
#     pose_analyzer = LungePoseAnalyzer()
#
#     for video_path, label in zip(video_paths, labels):
#         # 비디오에서 특징 추출
#         frames_features = pose_analyzer.process_lunge_video(video_path)
#
#         for features in frames_features:
#             features['label'] = label  # 레이블 추가
#             data.append(features)
#
#     return pd.DataFrame(data)
#
#
# class LungeDataset(Dataset):
#     def __init__(self, dataframe, feature_columns):
#         self.features = dataframe[feature_columns].values
#         self.labels = dataframe['label'].values
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         x = torch.tensor(self.features[idx], dtype=torch.float32)
#         y = torch.tensor(self.labels[idx], dtype=torch.long)
#         return x, y
#
#
# class LungeClassifier(nn.Module):
#     def __init__(self, input_size):
#         super(LungeClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.output = nn.Linear(32, 2)
#
#     def forward(self, x, get_probability=False):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.output(x)
#
#         if get_probability:
#             x=F.softmax(x,dim=1)
#         return x
#
#
# def main():
#     # 데이터 경로와 레이블 설정
#     # video_paths = ["augmented_videos/correct1_augmented_0.mp4", "data/incorrect1.mp4",
#     #                "data/런지자세.mp4", "data/incorrect2.mp4",
#     #                "data/correct3.mp4", "data/incorrect3.mp4",
#     #                "data/correct4.mp4", "data/incorrect4.mp4",
#     #                "data/correct5.mp4", "data/incorrect5.mp4",
#     #                "data/correct6.mp4", "data/incorrect6.mp4",
#     #                "data/correct7.mp4", "data/incorrect7.mp4",
#     #                "data/correct8.mp4", "data/incorrect8.mp4",
#     #                "data/correct9.mp4", "data/incorrect9.mp4",
#     #                "data/correct10.mp4", "data/incorrect10.mp4",
#     #                "data/correct11.mp4", "data/incorrect11.mp4"]
#     #
#     # labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
#
#     video_paths = []
#     labels = []
#
#     # Generate paths and labels for the correct videos
#     for i in range(1, 11):  # correct1 to correct4
#         for j in range(5):  # augmented_0 to augmented_4
#             video_paths.append(f"augmented_videos/correct{i}_augmented_{j}.mp4")
#             labels.append(1)
#
#     # Generate paths and labels for the incorrect videos
#     for i in range(1, 11):
#         for j in range(5):
#             video_paths.append(f"augmented_videos/incorrect{i}_augmented_{j}.mp4")
#             labels.append(0)
#
#     # 특징 추출
#     data_df = extract_features_from_video(video_paths, labels)
#
#     # 특징 컬럼 정의
#     feature_columns = ['right_knee_angle', 'left_knee_angle', 'right_hip_angle', 'left_hip_angle']
#     data_df = data_df[feature_columns + ['label']]
#
#     # Dataset 및 DataLoader 생성
#     dataset = LungeDataset(data_df, feature_columns)
#     train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
#
#     # 모델 및 학습 설정
#     model = LungeClassifier(input_size=len(feature_columns))
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     # 학습 루프
#     num_epochs = 1000
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
#
#     # 학습된 모델 저장
#     torch.save(model.state_dict(), "lunge_classifier.pth")


# class LungePoseAnalyzer:
#     def __init__(self):
#         self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         self.mp_pose = mp.solutions.pose
#         self.frames_per_point = 3  # 각 지점 전후로 선택할 프레임 수
#
#     def extract_features(self, video_path):
#         video = cv2.VideoCapture(video_path)
#         angles_by_frame = []
#         angles = {
#             'right_knee': [],
#             'left_knee': [],
#             'right_hip': [],
#             'left_hip': []
#         }
#
#         # 모든 프레임의 각도 저장
#         while video.isOpened():
#             ret, frame = video.read()
#             if not ret:
#                 break
#
#             results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#             if results.pose_landmarks:
#                 frame_angles = self.calculate_all_angles(results.pose_landmarks)
#                 angles_by_frame.append(frame_angles)
#                 for key in angles.keys():
#                     angles[key].append(frame_angles[key])
#
#         video.release()
#
#         if not angles_by_frame:
#             return []
#
#         # 주요 프레임 인덱스 계산
#         total_frames = len(angles_by_frame)
#
#         # 최저점 찾기
#         avg_angles = []
#         for i in range(len(angles_by_frame)):
#             avg_angle = sum(angles_by_frame[i].values()) / len(angles_by_frame[i])
#             avg_angles.append(avg_angle)
#         lowest_frame_idx = avg_angles.index(min(avg_angles))
#
#         # 중간점 계산
#         start_to_lowest_midpoint = lowest_frame_idx // 2
#         lowest_to_end_midpoint = (lowest_frame_idx + total_frames) // 2
#
#         # 선택할 프레임들의 인덱스와 라벨 정의
#         selected_frames = {
#             # 시작-최저점 중간지점 주변 (incorrect)
#             start_to_lowest_midpoint - 1: 0,
#             start_to_lowest_midpoint: 0,
#             start_to_lowest_midpoint + 1: 0,
#
#             # 최저점 직전 (correct)
#             lowest_frame_idx - 3: 1,
#             lowest_frame_idx - 2: 1,
#             lowest_frame_idx - 1: 1,
#
#             # 최저점 직후 (correct)
#             lowest_frame_idx + 1: 1,
#             lowest_frame_idx + 2: 1,
#             lowest_frame_idx + 3: 1,
#
#             # 최저점-끝점 중간지점 주변 (incorrect)
#             lowest_to_end_midpoint - 1: 0,
#             lowest_to_end_midpoint: 0,
#             lowest_to_end_midpoint + 1: 0,
#         }
#
#         # 결과 데이터 생성
#         features_list = []
#
#         # 선택된 프레임들만 처리
#         for frame_idx, label in selected_frames.items():
#             # 프레임 인덱스가 유효한 범위 내에 있는지 확인
#             if 0 <= frame_idx < len(angles_by_frame):
#                 frame_data = {
#                     'right_knee_angle': angles_by_frame[frame_idx]['right_knee'],
#                     'left_knee_angle': angles_by_frame[frame_idx]['left_knee'],
#                     'right_hip_angle': angles_by_frame[frame_idx]['right_hip'],
#                     'left_hip_angle': angles_by_frame[frame_idx]['left_hip'],
#                     'label': label,
#                     'frame_idx': frame_idx  # 디버깅을 위해 프레임 인덱스 추가
#                 }
#                 features_list.append(frame_data)
#
#         return features_list
#
#     def calculate_all_angles(self, landmarks):
#         # 기존 코드와 동일
#         right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
#         right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
#         right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
#
#         left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
#         left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
#         left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
#
#         right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
#         left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
#
#         angles = {
#             'right_knee': self.calculate_angle(right_hip, right_knee, right_ankle),
#             'left_knee': self.calculate_angle(left_hip, left_knee, left_ankle),
#             'right_hip': self.calculate_angle(right_shoulder, right_hip, right_knee),
#             'left_hip': self.calculate_angle(left_shoulder, left_hip, left_knee)
#         }
#
#         return angles
#
#     def calculate_angle(self, point1, point2, point3):
#         # 기존 코드와 동일
#         vector1 = np.array([point1.x - point2.x, point1.y - point2.y])
#         vector2 = np.array([point3.x - point2.x, point3.y - point2.y])
#
#         dot_product = np.dot(vector1, vector2)
#         magnitude1 = np.linalg.norm(vector1)
#         magnitude2 = np.linalg.norm(vector2)
#
#         cos_theta = dot_product / (magnitude1 * magnitude2)
#         angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
#
#         return np.degrees(angle)
#
#     def process_lunge_video(self, video_path):
#         return self.extract_features(video_path)

# if __name__ == "__main__":
#     main()

# import cv2
# import numpy as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import mediapipe as mp
# import pandas as pd
# from pathlib import Path
#
#
# class LungePoseAnalyzer:
#     def __init__(self):
#         self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         self.mp_pose = mp.solutions.pose
#         self.frames_per_point = 3
#
#     def extract_features(self, video_path):
#         video = cv2.VideoCapture(video_path)
#         angles_by_frame = []
#         angles = {
#             'right_knee': [],
#             'left_knee': [],
#             'right_hip': [],
#             'left_hip': []
#         }
#
#         while video.isOpened():
#             ret, frame = video.read()
#             if not ret:
#                 break
#
#             results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#             if results.pose_landmarks:
#                 frame_angles = self.calculate_all_angles(results.pose_landmarks)
#                 angles_by_frame.append(frame_angles)
#                 for key in angles.keys():
#                     angles[key].append(frame_angles[key])
#
#         video.release()
#
#         if not angles_by_frame:
#             return []
#
#         total_frames = len(angles_by_frame)
#
#         # Find lowest point
#         avg_angles = []
#         for i in range(len(angles_by_frame)):
#             avg_angle = sum(angles_by_frame[i].values()) / len(angles_by_frame[i])
#             avg_angles.append(avg_angle)
#         lowest_frame_idx = avg_angles.index(min(avg_angles))
#
#         # Calculate midpoints
#         start_to_lowest_midpoint = lowest_frame_idx // 2
#         lowest_to_end_midpoint = (lowest_frame_idx + total_frames) // 2
#
#         selected_frames = {
#             start_to_lowest_midpoint - 1: 0,
#             start_to_lowest_midpoint: 0,
#             start_to_lowest_midpoint + 1: 0,
#             lowest_frame_idx - 3: 1,
#             lowest_frame_idx - 2: 1,
#             lowest_frame_idx - 1: 1,
#             lowest_frame_idx + 1: 1,
#             lowest_frame_idx + 2: 1,
#             lowest_frame_idx + 3: 1,
#             lowest_to_end_midpoint - 1: 0,
#             lowest_to_end_midpoint: 0,
#             lowest_to_end_midpoint + 1: 0,
#         }
#
#         features_list = []
#         for frame_idx, frame_label in selected_frames.items():
#             if 0 <= frame_idx < len(angles_by_frame):
#                 frame_data = {
#                     'right_knee_angle': angles_by_frame[frame_idx]['right_knee'],
#                     'left_knee_angle': angles_by_frame[frame_idx]['left_knee'],
#                     'right_hip_angle': angles_by_frame[frame_idx]['right_hip'],
#                     'left_hip_angle': angles_by_frame[frame_idx]['left_hip'],
#                     'label': frame_label,  # using the frame-specific label
#                     'frame_idx': frame_idx
#                 }
#                 features_list.append(frame_data)
#
#         return features_list
#
#     def calculate_all_angles(self, landmarks):
#         right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
#         right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
#         right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
#
#         left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
#         left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
#         left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
#
#         right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
#         left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
#
#         angles = {
#             'right_knee': self.calculate_angle(right_hip, right_knee, right_ankle),
#             'left_knee': self.calculate_angle(left_hip, left_knee, left_ankle),
#             'right_hip': self.calculate_angle(right_shoulder, right_hip, right_knee),
#             'left_hip': self.calculate_angle(left_shoulder, left_hip, left_knee)
#         }
#
#         return angles
#
#     def calculate_angle(self, point1, point2, point3):
#         vector1 = np.array([point1.x - point2.x, point1.y - point2.y])
#         vector2 = np.array([point3.x - point2.x, point3.y - point2.y])
#
#         dot_product = np.dot(vector1, vector2)
#         magnitude1 = np.linalg.norm(vector1)
#         magnitude2 = np.linalg.norm(vector2)
#
#         cos_theta = dot_product / (magnitude1 * magnitude2)
#         angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
#
#         return np.degrees(angle)
#
#
# class LungeDataset(Dataset):
#     def __init__(self, data_df, feature_columns):
#         self.features = torch.FloatTensor(data_df[feature_columns].values)
#         self.labels = torch.LongTensor(data_df['label'].values)
#
#     def __len__(self):
#         return len(self.features)
#
#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]
#
#
# class LungeClassifier(nn.Module):
#     def __init__(self, input_size):
#         super(LungeClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 2)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# def extract_features_from_video(video_paths):
#     analyzer = LungePoseAnalyzer()
#     all_features = []
#
#     for video_path in video_paths:
#         features = analyzer.extract_features(video_path)
#         if features:
#             all_features.extend(features)  # 각 프레임의 특징과 라벨을 추가
#
#     return pd.DataFrame(all_features)
#
#
# def main():
#     # Generate video paths
#     video_paths = []
#
#     # Generate paths for correct videos
#     for i in range(1, 11):
#         for j in range(5):
#             video_paths.append(f"augmented_videos/correct{i}_augmented_{j}.mp4")
#
#     # Generate paths for incorrect videos
#     for i in range(1, 11):
#         for j in range(5):
#             video_paths.append(f"augmented_videos/incorrect{i}_augmented_{j}.mp4")
#
#     # Extract features with frame-specific labels
#     data_df = extract_features_from_video(video_paths)
#
#     # Define feature columns
#     feature_columns = ['right_knee_angle', 'left_knee_angle', 'right_hip_angle', 'left_hip_angle']
#     data_df = data_df[feature_columns + ['label']]
#
#     # Print class distribution
#     print("Class distribution:")
#     print(data_df['label'].value_counts())
#
#     # Create dataset and dataloader
#     dataset = LungeDataset(data_df, feature_columns)
#     train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
#     # Initialize model and training parameters
#     model = LungeClassifier(input_size=len(feature_columns))
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     # Training loop
#     num_epochs = 5000
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         if (epoch + 1) % 100 == 0:
#             accuracy = 100 * correct / total
#             print(f"Epoch {epoch + 1}/{num_epochs}")
#             print(f"Loss: {running_loss / len(train_loader):.4f}")
#             print(f"Accuracy: {accuracy:.2f}%")
#
#     # Save the trained model
#     torch.save(model.state_dict(), "lunge_classifier.pth")
#     print("Training completed and model saved!")
#
#
# if __name__ == "__main__":
#     main()


import os
import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class LungePoseAnalyzer:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.correct_count = 0
        self.incorrect_count = 0

    def extract_features(self, video_path):
        video = cv2.VideoCapture(video_path)
        angles_by_frame = []
        angles = {
            'right_knee': [],
            'left_knee': [],
            'right_hip': [],
            'left_hip': []
        }

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                frame_angles = self.calculate_all_angles(results.pose_landmarks)
                angles_by_frame.append(frame_angles)

        video.release()
        if not angles_by_frame:
            return []

        avg_angles = [sum(frame.values()) / len(frame) for frame in angles_by_frame]
        lowest_frame_idx = avg_angles.index(min(avg_angles))

        if video_path.split('/')[-1].startswith("correct"):
            features = self.label_correct_frames(angles_by_frame, len(angles_by_frame), lowest_frame_idx)
            self.correct_count += len(features)  # Correct 라벨링 데이터 수 추가
            return features
        elif video_path.split('/')[-1].startswith("incorrect"):
            features = self.label_incorrect_frames(angles_by_frame, lowest_frame_idx)
            self.incorrect_count += len(features)  # Incorrect 라벨링 데이터 수 추가
            return features
        else:
            return []

    def calculate_all_angles(self, landmarks):
        def calculate_angle(a, b, c):
            a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            return np.abs(radians * 180.0 / np.pi) % 360

        return {
            'right_knee': calculate_angle(landmarks.landmark[24], landmarks.landmark[26], landmarks.landmark[28]),
            'left_knee': calculate_angle(landmarks.landmark[23], landmarks.landmark[25], landmarks.landmark[27]),
            'right_hip': calculate_angle(landmarks.landmark[12], landmarks.landmark[24], landmarks.landmark[26]),
            'left_hip': calculate_angle(landmarks.landmark[11], landmarks.landmark[23], landmarks.landmark[25]),
        }

    def label_correct_frames(self, angles_by_frame, total_frames, lowest_frame_idx):
        start_to_lowest_midpoint = lowest_frame_idx // 2
        lowest_to_end_midpoint = (lowest_frame_idx + total_frames) // 2

        selected_frames = {  # 중간지점 및 최소 프레임 중심
            start_to_lowest_midpoint - 1: 0,
            start_to_lowest_midpoint: 0,
            start_to_lowest_midpoint + 1: 0,
            lowest_frame_idx - 3: 1,
            lowest_frame_idx - 2: 1,
            lowest_frame_idx - 1: 1,
            lowest_frame_idx + 1: 1,
            lowest_frame_idx + 2: 1,
            lowest_frame_idx + 3: 1,
            lowest_to_end_midpoint - 1: 0,
            lowest_to_end_midpoint: 0,
            lowest_to_end_midpoint + 1: 0,
        }
        return self.prepare_features(angles_by_frame, selected_frames)

    def label_incorrect_frames(self, angles_by_frame, lowest_frame_idx):
        selected_frames = {
            lowest_frame_idx - 2: 0,
            lowest_frame_idx - 1: 0,
            lowest_frame_idx : 0,
            lowest_frame_idx + 1: 0,
            lowest_frame_idx + 2: 0
        }
        return self.prepare_features(angles_by_frame, selected_frames)

    def prepare_features(self, angles_by_frame, selected_frames):
        features_list = []
        for frame_idx, frame_label in selected_frames.items():
            if 0 <= frame_idx < len(angles_by_frame):
                frame_data = {
                    'right_knee_angle': angles_by_frame[frame_idx]['right_knee'],
                    'left_knee_angle': angles_by_frame[frame_idx]['left_knee'],
                    'right_hip_angle': angles_by_frame[frame_idx]['right_hip'],
                    'left_hip_angle': angles_by_frame[frame_idx]['left_hip'],
                    'label': frame_label
                }
                features_list.append(frame_data)
                self.show_frame(frame_idx, frame_data, frame_label)
        return features_list

    def show_frame(self, frame_idx, frame_data, label):
        print(f"Frame {frame_idx} labeled as {'Correct' if label == 1 else 'Incorrect'}:")
        print(frame_data)

    def print_label_counts(self):
        print(f"Correct labeled data count: {self.correct_count}")
        print(f"Incorrect labeled data count: {self.incorrect_count}")


### **2. 데이터셋 정의 및 학습 코드**
class LungeDataset(Dataset):
    def __init__(self, data):
        self.data = [torch.tensor([frame['right_knee_angle'], frame['left_knee_angle'], frame['right_hip_angle'], frame['left_hip_angle']], dtype=torch.float32) for frame in data]
        self.labels = torch.tensor([frame['label'] for frame in data], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 모델 정의
class LungeClassifier(nn.Module):
    def __init__(self):
        super(LungeClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(data, num_epochs=2000):
    dataset = LungeDataset(data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LungeClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        total_loss, total_correct = 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs} Loss: {total_loss:.4f}, Accuracy: {100 * total_correct / len(dataset):.2f}%')

    torch.save(model.state_dict(), 'lunge_classifier.pth')


### **3. 실행 예시 (main 함수 예시)**
def main():
    # LungePoseAnalyzer 객체 생성
    analyzer = LungePoseAnalyzer()

    # 비디오 파일 경로들 (올바른 비디오와 잘못된 비디오 포함)
    video_paths = []
    for i in range(1, 16):
        for j in range(5):  # augmented_0 to augmented_4
            video_paths.append(f"augmented_videos/correct{i}_augmented_{j}.mp4")
            video_paths.append(f"augmented_videos/incorrect{i}_augmented_{j}.mp4")

    # 각 비디오에 대해 특징을 추출하고 라벨링
    all_data = []
    for video_path in video_paths:
        features = analyzer.extract_features(video_path)
        all_data.extend(features)  # 모든 비디오의 데이터를 모음

    # # 라벨링 된 데이터 개수 출력
    # analyzer.print_label_counts()

    # # 각 비디오에 대해 특징을 추출하고 라벨링
    # for video_path in video_paths:
    #     features, correct_count, incorrect_count = analyzer.extract_features(video_path)
    #     print(f"Video {video_path} - Correct frames: {correct_count}, Incorrect frames: {incorrect_count}")

    # 라벨링 된 데이터 개수 출력
    analyzer.print_label_counts()

    # 학습을 위한 모델 훈련
    if all_data:
        print("Training the model...")
        train_model(all_data)  # 학습 함수 호출



if __name__ == "__main__":
    main()