# def analyze_form(result):
#     """자세 분석 및 피드백 제공"""
#     feedback = []
#
#     # 'Correct'일 경우 자세 피드백을 생략
#     if result['prediction'] == 'Correct':
#         return ["전반적으로 좋은 자세입니다!"]
#
#     # 무릎 각도 분석
#     knee_threshold = 90  # 적절한 무릎 각도 임계값
#     if result['angles']['right_knee'] > knee_threshold:
#         feedback.append("오른쪽 무릎을 더 굽혀주세요.")
#     if result['angles']['left_knee'] > knee_threshold:
#         feedback.append("왼쪽 무릎을 더 굽혀주세요.")
#
#     # 엉덩이 각도 분석
#     hip_threshold = 120  # 적절한 엉덩이 각도 임계값
#     if result['angles']['right_hip'] > hip_threshold:
#         feedback.append("오른쪽 엉덩이를 더 낮춰주세요.")
#     if result['angles']['left_hip'] > hip_threshold:
#         feedback.append("왼쪽 엉덩이를 더 낮춰주세요.")
#
#     # 피드백이 없으면 올바른 자세라는 메시지 추가
#     if not feedback:
#         feedback.append("전반적으로 좋은 자세입니다!")
#
#     return feedback
#
# import torch
# import torch.nn.functional as F
# import pandas as pd
#
# from lunge_anaylzer import LungePoseAnalyzer, LungeClassifier
# def test_new_video(video_path, model_path="lunge_classifier.pth", confidence_threshold=0.8):
#     # LungePoseAnalyzer 인스턴스 생성
#     pose_analyzer = LungePoseAnalyzer()
#
#     # 비디오에서 특징 추출
#     features = pose_analyzer.extract_features(video_path)
#
#     # 특징을 DataFrame으로 변환
#     feature_columns = ['right_knee_angle', 'left_knee_angle', 'right_hip_angle', 'left_hip_angle']
#     test_data = pd.DataFrame([features])
#
#     # 모델 로드
#     model = LungeClassifier(input_size=len(feature_columns))
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#
#     # 데이터 전처리
#     test_features = torch.tensor(test_data[feature_columns].values, dtype=torch.float32)
#
#     # 예측
#     with torch.no_grad():
#         # 소프트맥스를 적용하지 않은 로짓 값 출력
#         outputs = model(test_features)
#         print("로짓 값 (소프트맥스 전):", outputs.numpy())
#
#         # 소프트맥스를 적용하여 확률로 변환
#         probabilities = F.softmax(outputs, dim=1).numpy()
#         print("소프트맥스 적용 후 확률:", probabilities)
#
#         # "Incorrect" 확률 (클래스 0)과 "Correct" 확률 (클래스 1)
#         incorrect_confidence = probabilities[0][0]
#         correct_confidence = probabilities[0][1]
#
#         # 임계값을 기준으로 판정
#         predicted_class = 'Correct' if correct_confidence >= confidence_threshold else 'Incorrect'
#
#     # 결과 분석
#     result = {
#         'prediction': predicted_class,
#         'confidence': float(correct_confidence if predicted_class == 'Correct' else incorrect_confidence),
#         'incorrect_confidence': float(incorrect_confidence),
#         'correct_confidence': float(correct_confidence),
#         'angles': {
#             'right_knee': features['right_knee_angle'],
#             'left_knee': features['left_knee_angle'],
#             'right_hip': features['right_hip_angle'],
#             'left_hip': features['left_hip_angle']
#         }
#     }
#
#     return result
# def main_test():
#     # 테스트할 비디오 경로
#     test_video_path = "data/lunge_test_correct1_augmented_0.mp4"
#
#     try:
#         # 비디오 분석
#         print("비디오 분석 중...")
#         result = test_new_video(test_video_path)
#
#         # 결과 출력
#         print("\n=== 런지 동작 분석 결과 ===")
#         print(f"판정: {result['prediction']}")
#         print(f"신뢰도 (Correct): {result['correct_confidence']:.2%}")
#         print(f"Incorrect에 대한 확신도: {result['incorrect_confidence']:.2%}")
#
#         print("\n=== 관절 각도 ===")
#         for joint, angle in result['angles'].items():
#             print(f"{joint}: {angle:.1f}도")
#
#         print("\n=== 자세 피드백 ===")
#         feedback = analyze_form(result)
#         for fb in feedback:
#             print(f"- {fb}")
#
#     except Exception as e:
#         print(f"에러 발생: {str(e)}")
#
#
# if __name__ == "__main__":
#     main_test()

# import torch
# import cv2
#
#
# class LungeModelTester:
#     def __init__(self, model_path):
#         # 모델 로드
#         self.model = torch.load(model_path)
#         self.model.eval()  # 평가 모드로 설정
#
#     def process_video(self, video_path):
#         video = cv2.VideoCapture(video_path)
#         angles_by_frame = []  # 각 프레임에서의 각도를 저장할 리스트
#
#         while video.isOpened():
#             ret, frame = video.read()
#             if not ret:
#                 break
#
#             # 포즈 추출 (예시로 PoseDetector를 사용한다고 가정)
#             results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             if results.pose_landmarks:
#                 frame_angles = self.calculate_all_angles(results.pose_landmarks)
#                 angles_by_frame.append(frame_angles)  # 각도를 각 프레임에 대해 추출하여 저장
#
#         video.release()
#
#         if not angles_by_frame:
#             return []
#
#         # 각도 평균 계산 후 최저점 프레임 찾기
#         avg_angles = [sum(frame.values()) / len(frame) for frame in angles_by_frame]
#         lowest_frame_idx = avg_angles.index(min(avg_angles))  # 최저점 프레임 인덱스
#
#         return lowest_frame_idx, angles_by_frame
#
#     def label_video(self, video_path):
#         lowest_frame_idx, angles_by_frame = self.process_video(video_path)
#
#         # 모델에 입력할 데이터 준비
#         frame_data = self.prepare_frame_data(angles_by_frame, lowest_frame_idx)
#
#         # 모델을 통해 예측
#         prediction = self.predict_with_model(frame_data)
#
#         # 예측된 라벨을 바탕으로 올바른 라벨인지 판별
#         if prediction == 1:
#             print(f"Video {video_path}: Correct label")
#         else:
#             print(f"Video {video_path}: Incorrect label")
#
#     def prepare_frame_data(self, angles_by_frame, lowest_frame_idx):
#         # 최저점에 해당하는 프레임만 모델에 맞게 전처리
#         frame_data = angles_by_frame[lowest_frame_idx]  # 최저점 프레임의 각도 데이터
#         return frame_data  # 모델 입력 형태로 변환 (필요에 따라 추가 전처리)
#
#     def predict_with_model(self, frame_data):
#         # 모델을 통해 예측 (간단한 예시)
#         inputs = torch.tensor(frame_data).float()  # 데이터를 텐서로 변환
#         outputs = self.model(inputs)  # 모델을 통해 예측
#         _, predicted = torch.max(outputs, 1)  # 가장 높은 확률의 라벨을 선택
#         return predicted.item()
#
#
# # 모델 경로 설정
# model_path = "lunge_classifier.pth"
# tester = LungeModelTester(model_path)
#
# # 비디오 파일 경로 (예시)
# video_paths = []
# for i in range(1, 10):
#     for j in range(5):  # augmented_0 to augmented_4
#         video_paths.append(f"augmented_videos/lunge_test_correct{i}_augmented_{j}.mp4")
#
# # 각 비디오에 대해 테스트 수행
# for video_path in video_paths:
#     tester.label_video(video_path)

# import torch
# import cv2
# import numpy as np
#
#
# class LungeModelTester:
#     def __init__(self, model_path):
#         # 모델 로드
#         self.model = torch.load(model_path)
#         self.model.eval()  # 평가 모드로 설정
#
#         # 포즈 추출에 사용할 PoseDetector 초기화 (여기서는 예시로 'pose'라는 이름을 사용)
#         self.pose = PoseDetector()  # PoseDetector는 실제 포즈 추출 라이브러리로 대체 필요
#
#     def process_video(self, video_path):
#         video = cv2.VideoCapture(video_path)
#         angles_by_frame = []  # 각 프레임에서의 각도를 저장할 리스트
#
#         while video.isOpened():
#             ret, frame = video.read()
#             if not ret:
#                 break
#
#             # 포즈 추출 (예시로 PoseDetector를 사용한다고 가정)
#             results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             if results.pose_landmarks:
#                 frame_angles = self.calculate_all_angles(results.pose_landmarks)
#                 angles_by_frame.append(frame_angles)  # 각도를 각 프레임에 대해 추출하여 저장
#
#         video.release()
#
#         if not angles_by_frame:
#             return []
#
#         # 각도 평균 계산 후 최저점 프레임 찾기
#         avg_angles = [sum(frame.values()) / len(frame) for frame in angles_by_frame]
#         lowest_frame_idx = avg_angles.index(min(avg_angles))  # 최저점 프레임 인덱스
#
#         return lowest_frame_idx, angles_by_frame
#
#     def label_video(self, video_path):
#         lowest_frame_idx, angles_by_frame = self.process_video(video_path)
#
#         if lowest_frame_idx is None or not angles_by_frame:
#             print(f"Video {video_path}: No valid frames for labeling.")
#             return
#
#         # 모델에 입력할 데이터 준비
#         frame_data = self.prepare_frame_data(angles_by_frame, lowest_frame_idx)
#
#         # 모델을 통해 예측
#         prediction = self.predict_with_model(frame_data)
#
#         # 예측된 라벨을 바탕으로 올바른 라벨인지 판별
#         if prediction == 1:
#             print(f"Video {video_path}: Correct label")
#         else:
#             print(f"Video {video_path}: Incorrect label")
#
#     def prepare_frame_data(self, angles_by_frame, lowest_frame_idx):
#         # 최저점에 해당하는 프레임만 모델에 맞게 전처리
#         frame_data = angles_by_frame[lowest_frame_idx]  # 최저점 프레임의 각도 데이터
#         return frame_data  # 모델 입력 형태로 변환 (필요에 따라 추가 전처리)
#
#     def predict_with_model(self, frame_data):
#         # 모델을 통해 예측 (간단한 예시)
#         inputs = torch.tensor(frame_data).float()  # 데이터를 텐서로 변환
#         inputs = inputs.unsqueeze(0)  # 배치 차원 추가 (모델 입력 형식에 맞게)
#         outputs = self.model(inputs)  # 모델을 통해 예측
#         _, predicted = torch.max(outputs, 1)  # 가장 높은 확률의 라벨을 선택
#         return predicted.item()
#
#     def calculate_all_angles(self, pose_landmarks):
#         # 포즈 랜드마크에서 필요한 각도를 계산하는 함수 (예시)
#         # 실제 포즈 추출 라이브러리에서 각도를 계산하는 코드 필요
#         angles = {}
#         # 예시로 두 무릎과 두 엉덩이 각도를 계산한다고 가정
#         angles['right_knee'] = self.calculate_angle(pose_landmarks, 'right_knee')
#         angles['left_knee'] = self.calculate_angle(pose_landmarks, 'left_knee')
#         angles['right_hip'] = self.calculate_angle(pose_landmarks, 'right_hip')
#         angles['left_hip'] = self.calculate_angle(pose_landmarks, 'left_hip')
#         return angles
#
#     def calculate_angle(self, landmarks, joint_name):
#         # 실제로는 포즈 랜드마크에서 각도를 계산하는 로직이 필요함
#         return np.random.random() * 90  # 임시로 무작위 각도를 반환
#
#
# # 모델 경로 설정
# model_path = "lunge_classifier.pth"
# tester = LungeModelTester(model_path)
#
# # 비디오 파일 경로 (예시)
# video_paths = []
# for i in range(1, 10):
#     for j in range(5):  # augmented_0 to augmented_4
#         video_paths.append(f"augmented_videos/lunge_test_correct{i}_augmented_{j}.mp4")
#
# # 각 비디오에 대해 테스트 수행
# for video_path in video_paths:
#     tester.label_video(video_path)

# import torch
# import cv2
# import numpy as np
#
#
# class LungeModelTester:
#     def __init__(self, model_path):
#         # 모델 클래스 정의 (LungePoseModel로 가정)
#         self.model = LungePoseModel()  # 실제 모델 클래스에 맞게 정의
#         checkpoint = torch.load(model_path)
#
#         # 모델이 state_dict 형태로 저장되어 있다면, 이를 로드
#         self.model.load_state_dict(checkpoint)
#
#         self.model.eval()  # 평가 모드로 설정
#
#         # 포즈 추출에 사용할 PoseDetector 초기화 (여기서는 예시로 'pose'라는 이름을 사용)
#         self.pose = PoseDetector()  # PoseDetector는 실제 포즈 추출 라이브러리로 대체 필요
#
#     def process_video(self, video_path):
#         video = cv2.VideoCapture(video_path)
#         angles_by_frame = []  # 각 프레임에서의 각도를 저장할 리스트
#
#         while video.isOpened():
#             ret, frame = video.read()
#             if not ret:
#                 break
#
#             # 포즈 추출 (예시로 PoseDetector를 사용한다고 가정)
#             results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             if results.pose_landmarks:
#                 frame_angles = self.calculate_all_angles(results.pose_landmarks)
#                 angles_by_frame.append(frame_angles)  # 각도를 각 프레임에 대해 추출하여 저장
#
#         video.release()
#
#         if not angles_by_frame:
#             return []
#
#         # 각도 평균 계산 후 최저점 프레임 찾기
#         avg_angles = [sum(frame.values()) / len(frame) for frame in angles_by_frame]
#         lowest_frame_idx = avg_angles.index(min(avg_angles))  # 최저점 프레임 인덱스
#
#         return lowest_frame_idx, angles_by_frame
#
#     def label_video(self, video_path):
#         lowest_frame_idx, angles_by_frame = self.process_video(video_path)
#
#         if lowest_frame_idx is None or not angles_by_frame:
#             print(f"Video {video_path}: No valid frames for labeling.")
#             return
#
#         # 모델에 입력할 데이터 준비
#         frame_data = self.prepare_frame_data(angles_by_frame, lowest_frame_idx)
#
#         # 모델을 통해 예측
#         prediction = self.predict_with_model(frame_data)
#
#         # 예측된 라벨을 바탕으로 올바른 라벨인지 판별
#         if prediction == 1:
#             print(f"Video {video_path}: Correct label")
#         else:
#             print(f"Video {video_path}: Incorrect label")
#
#     def prepare_frame_data(self, angles_by_frame, lowest_frame_idx):
#         # 최저점에 해당하는 프레임만 모델에 맞게 전처리
#         frame_data = angles_by_frame[lowest_frame_idx]  # 최저점 프레임의 각도 데이터
#         return frame_data  # 모델 입력 형태로 변환 (필요에 따라 추가 전처리)
#
#     def predict_with_model(self, frame_data):
#         # 모델을 통해 예측 (간단한 예시)
#         inputs = torch.tensor(frame_data).float()  # 데이터를 텐서로 변환
#         inputs = inputs.unsqueeze(0)  # 배치 차원 추가 (모델 입력 형식에 맞게)
#         outputs = self.model(inputs)  # 모델을 통해 예측
#         _, predicted = torch.max(outputs, 1)  # 가장 높은 확률의 라벨을 선택
#         return predicted.item()
#
#     def calculate_all_angles(self, pose_landmarks):
#         # 포즈 랜드마크에서 필요한 각도를 계산하는 함수 (예시)
#         # 실제 포즈 추출 라이브러리에서 각도를 계산하는 코드 필요
#         angles = {}
#         # 예시로 두 무릎과 두 엉덩이 각도를 계산한다고 가정
#         angles['right_knee'] = self.calculate_angle(pose_landmarks, 'right_knee')
#         angles['left_knee'] = self.calculate_angle(pose_landmarks, 'left_knee')
#         angles['right_hip'] = self.calculate_angle(pose_landmarks, 'right_hip')
#         angles['left_hip'] = self.calculate_angle(pose_landmarks, 'left_hip')
#         return angles
#
#     def calculate_angle(self, landmarks, joint_name):
#         # 실제로는 포즈 랜드마크에서 각도를 계산하는 로직이 필요함
#         return np.random.random() * 90  # 임시로 무작위 각도를 반환
#
#
# # 모델 경로 설정
# model_path = "lunge_classifier.pth"
# tester = LungeModelTester(model_path)
#
# # 비디오 파일 경로 (예시)
# video_paths = []
# for i in range(1, 10):
#     for j in range(5):  # augmented_0 to augmented_4
#         video_paths.append(f"augmented_videos/lunge_test_correct{i}_augmented_{j}.mp4")
#
# # 각 비디오에 대해 테스트 수행
# for video_path in video_paths:
#     tester.label_video(video_path)

# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# import mediapipe as mp
#
#
# class LungePoseModel(nn.Module):
#     def __init__(self):
#         super(LungePoseModel, self).__init__()
#         # Simple neural network for pose classification
#         self.layers = nn.Sequential(
#             nn.Linear(8, 64),  # 8 input features (4 joints × 2 angles)
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2)  # 2 classes: correct/incorrect
#         )
#
#     def forward(self, x):
#         return self.layers(x)
#
#
# class PoseDetector:
#     def __init__(self):
#         self.mp_pose = mp.solutions.pose
#         self.pose = self.mp_pose.Pose(
#             static_image_mode=False,
#             model_complexity=1,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#
#     def process(self, image):
#         return self.pose.process(image)
#
#     def get_landmarks(self, results):
#         if results.pose_landmarks:
#             return results.pose_landmarks.landmark
#         return None
#
#
# class LungeModelTester:
#     def __init__(self, model_path):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = LungePoseModel().to(self.device)
#
#         try:
#             checkpoint = torch.load(model_path, map_location=self.device)
#             self.model.load_state_dict(checkpoint)
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             raise
#
#         self.model.eval()
#         self.pose = PoseDetector()
#
#     def process_video(self, video_path):
#         try:
#             video = cv2.VideoCapture(video_path)
#             if not video.isOpened():
#                 raise ValueError(f"Could not open video file: {video_path}")
#
#             angles_by_frame = []
#
#             while video.isOpened():
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#
#                 results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                 if results.pose_landmarks:
#                     frame_angles = self.calculate_all_angles(results.pose_landmarks)
#                     angles_by_frame.append(frame_angles)
#
#             video.release()
#
#             if not angles_by_frame:
#                 return None, []
#
#             # Find lowest position frame
#             avg_angles = [sum(frame.values()) / len(frame) for frame in angles_by_frame]
#             lowest_frame_idx = avg_angles.index(min(avg_angles))
#
#             return lowest_frame_idx, angles_by_frame
#
#         except Exception as e:
#             print(f"Error processing video {video_path}: {e}")
#             return None, []
#
#     def calculate_all_angles(self, pose_landmarks):
#         landmarks = self.pose.get_landmarks(pose_landmarks)
#         if not landmarks:
#             return {}
#
#         angles = {
#             'right_knee': self.calculate_knee_angle(landmarks, 'right'),
#             'left_knee': self.calculate_knee_angle(landmarks, 'left'),
#             'right_hip': self.calculate_hip_angle(landmarks, 'right'),
#             'left_hip': self.calculate_hip_angle(landmarks, 'left')
#         }
#         return angles
#
#     def calculate_knee_angle(self, landmarks, side):
#         if side == 'right':
#             hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
#             knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
#             ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
#         else:
#             hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
#             knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
#             ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
#
#         return self.calculate_angle_3d(hip, knee, ankle)
#
#     def calculate_hip_angle(self, landmarks, side):
#         if side == 'right':
#             shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
#             hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
#             knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
#         else:
#             shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
#             hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
#             knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
#
#         return self.calculate_angle_3d(shoulder, hip, knee)
#
#     def calculate_angle_3d(self, point1, point2, point3):
#         # Convert landmarks to numpy arrays
#         a = np.array([point1.x, point1.y, point1.z])
#         b = np.array([point2.x, point2.y, point2.z])
#         c = np.array([point3.x, point3.y, point3.z])
#
#         # Calculate vectors
#         ba = a - b
#         bc = c - b
#
#         # Calculate angle using dot product
#         cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#         angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#
#         return np.degrees(angle)
#
#     def prepare_frame_data(self, angles_by_frame, lowest_frame_idx):
#         frame_data = angles_by_frame[lowest_frame_idx]
#         # Convert dictionary to list in a consistent order
#         features = [
#             frame_data['right_knee'], frame_data['left_knee'],
#             frame_data['right_hip'], frame_data['left_hip']
#         ]
#         return torch.tensor(features, dtype=torch.float32).to(self.device)
#
#     def predict_with_model(self, frame_data):
#         with torch.no_grad():
#             outputs = self.model(frame_data.unsqueeze(0))
#             _, predicted = torch.max(outputs, 1)
#         return predicted.item()
#
#     def label_video(self, video_path):
#         try:
#             lowest_frame_idx, angles_by_frame = self.process_video(video_path)
#
#             if lowest_frame_idx is None or not angles_by_frame:
#                 print(f"Video {video_path}: No valid frames for labeling.")
#                 return
#
#             frame_data = self.prepare_frame_data(angles_by_frame, lowest_frame_idx)
#             prediction = self.predict_with_model(frame_data)
#
#             result = "Correct" if prediction == 1 else "Incorrect"
#             print(f"Video {video_path}: {result} form")
#
#         except Exception as e:
#             print(f"Error processing {video_path}: {e}")
#
#
# def main():
#     # Model path configuration
#     model_path = "lunge_classifier.pth"
#
#     try:
#         # Initialize tester
#         tester = LungeModelTester(model_path)
#
#         # Generate video paths
#         video_paths = [
#             f"augmented_videos/lunge_test_correct{i}_augmented_{j}.mp4"
#             for i in range(1, 10)
#             for j in range(5)
#         ]
#
#         # Process each video
#         for video_path in video_paths:
#             tester.label_video(video_path)
#
#     except Exception as e:
#         print(f"Error in main execution: {e}")
#
#
# if __name__ == "__main__":
#     main()

# import torch
# import cv2
# import numpy as np
# import mediapipe as mp
#
#
# class PoseDetector:
#     def __init__(self):
#         self.mp_pose = mp.solutions.pose
#         self.pose = self.mp_pose.Pose(
#             static_image_mode=False,
#             model_complexity=1,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#
#     def find_pose(self, img):
#         results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         return results
#
#     def get_position(self, results, img):
#         landmarks = []
#         if results.pose_landmarks:
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 landmarks.append([id, cx, cy])
#         return landmarks
#
#     def calculate_angle(self, point1, point2, point3):
#         # 세 점 사이의 각도 계산
#         a = np.array(point1)
#         b = np.array(point2)
#         c = np.array(point3)
#
#         ba = a - b
#         bc = c - b
#
#         cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#         angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#
#         return np.degrees(angle)
#
#
# def test_lunge_video(video_path, model_path):
#     # GPU 사용 가능시 GPU 사용
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 저장된 모델 로드
#     model = torch.load(model_path, map_location=device)
#     model.eval()
#
#     # 비디오 캡처 객체 생성
#     cap = cv2.VideoCapture(video_path)
#     detector = PoseDetector()
#
#     # 각 프레임별 무릎 각도를 저장할 리스트
#     knee_angles = []
#     frame_landmarks = []
#
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#
#         # 포즈 감지
#         results = detector.find_pose(img)
#         landmarks = detector.get_position(results, img)
#
#         if landmarks:
#             # 오른쪽 무릎 각도 계산 (힙, 무릎, 발목)
#             right_hip = landmarks[24][:2]  # x, y 좌표만 사용
#             right_knee = landmarks[26][:2]
#             right_ankle = landmarks[28][:2]
#
#             # 왼쪽 무릎 각도 계산
#             left_hip = landmarks[23][:2]
#             left_knee = landmarks[25][:2]
#             left_ankle = landmarks[27][:2]
#
#             # 각도 계산
#             right_knee_angle = detector.calculate_angle(right_hip, right_knee, right_ankle)
#             left_knee_angle = detector.calculate_angle(left_hip, left_knee, left_ankle)
#
#             # 각도와 랜드마크 저장
#             knee_angles.append((right_knee_angle + left_knee_angle) / 2)  # 양쪽 무릎 각도의 평균
#             frame_landmarks.append(landmarks)
#
#     cap.release()
#
#     if not knee_angles:
#         return "No pose detected in video"
#
#     # 최저점 프레임 찾기 (무릎 각도가 가장 작은 지점)
#     lowest_frame_idx = knee_angles.index(min(knee_angles))
#
#     # 최저점 프레임의 특징 추출
#     lowest_frame_landmarks = frame_landmarks[lowest_frame_idx]
#
#     # 모델 입력을 위한 특징 준비
#     # 예: 주요 관절 각도들을 특징으로 사용
#     features = []
#
#     # 오른쪽 무릎 각도
#     right_hip = lowest_frame_landmarks[24][:2]
#     right_knee = lowest_frame_landmarks[26][:2]
#     right_ankle = lowest_frame_landmarks[28][:2]
#     right_knee_angle = detector.calculate_angle(right_hip, right_knee, right_ankle)
#     features.append(right_knee_angle)
#
#     # 왼쪽 무릎 각도
#     left_hip = lowest_frame_landmarks[23][:2]
#     left_knee = lowest_frame_landmarks[25][:2]
#     left_ankle = lowest_frame_landmarks[27][:2]
#     left_knee_angle = detector.calculate_angle(left_hip, left_knee, left_ankle)
#     features.append(left_knee_angle)
#
#     # 오른쪽 힙 각도 (어깨, 힙, 무릎)
#     right_shoulder = lowest_frame_landmarks[12][:2]
#     right_hip_angle = detector.calculate_angle(right_shoulder, right_hip, right_knee)
#     features.append(right_hip_angle)
#
#     # 왼쪽 힙 각도
#     left_shoulder = lowest_frame_landmarks[11][:2]
#     left_hip_angle = detector.calculate_angle(left_shoulder, left_hip, left_knee)
#     features.append(left_hip_angle)
#
#     # 특징을 텐서로 변환
#     features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
#
#     # 모델 예측
#     with torch.no_grad():
#         outputs = model(features_tensor.unsqueeze(0))
#         _, predicted = torch.max(outputs, 1)
#
#     # 결과 반환 (0: incorrect, 1: correct)
#     return "Correct form" if predicted.item() == 1 else "Incorrect form"
#
#
# def main():
#     model_path = "lunge_classifier.pth"
#
#     # 테스트할 비디오 경로 리스트 생성
#     video_paths = [
#         f"augmented_videos/lunge_test_correct{i}_augmented_{j}.mp4"
#         for i in range(1, 10)
#         for j in range(5)
#     ]
#
#     # 각 비디오 테스트
#     for video_path in video_paths:
#         try:
#             result = test_lunge_video(video_path, model_path)
#             print(f"Video {video_path}: {result}")
#         except Exception as e:
#             print(f"Error processing {video_path}: {e}")
#
#
# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
import cv2
import numpy as np
import mediapipe as mp


# 모델 클래스 정의
# class LungePoseModel(nn.Module):
#     def __init__(self):
#         super(LungePoseModel, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(4, 64),  # 입력: 4개의 각도 특징
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2)  # 출력: correct/incorrect 2개 클래스
#         )
#
#     def forward(self, x):
#         return self.classifier(x)
#
#
# class PoseDetector:
#     def __init__(self):
#         self.mp_pose = mp.solutions.pose
#         self.pose = self.mp_pose.Pose(
#             static_image_mode=False,
#             model_complexity=1,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#
#     def find_pose(self, img):
#         results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         return results
#
#     def get_position(self, results, img):
#         landmarks = []
#         if results.pose_landmarks:
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 landmarks.append([id, cx, cy])
#         return landmarks
#
#     def calculate_angle(self, point1, point2, point3):
#         a = np.array(point1)
#         b = np.array(point2)
#         c = np.array(point3)
#
#         ba = a - b
#         bc = c - b
#
#         cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#         angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#
#         return np.degrees(angle)
#
#
# def test_lunge_video(video_path, model_path):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 모델 초기화 및 가중치 로드
#     model = LungePoseModel()
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#
#     cap = cv2.VideoCapture(video_path)
#     detector = PoseDetector()
#
#     knee_angles = []
#     frame_landmarks = []
#
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#
#         results = detector.find_pose(img)
#         landmarks = detector.get_position(results, img)
#
#         if landmarks:
#             right_hip = landmarks[24][:2]
#             right_knee = landmarks[26][:2]
#             right_ankle = landmarks[28][:2]
#
#             left_hip = landmarks[23][:2]
#             left_knee = landmarks[25][:2]
#             left_ankle = landmarks[27][:2]
#
#             right_knee_angle = detector.calculate_angle(right_hip, right_knee, right_ankle)
#             left_knee_angle = detector.calculate_angle(left_hip, left_knee, left_ankle)
#
#             knee_angles.append((right_knee_angle + left_knee_angle) / 2)
#             frame_landmarks.append(landmarks)
#
#     cap.release()
#
#     if not knee_angles:
#         return "No pose detected in video"
#
#     lowest_frame_idx = knee_angles.index(min(knee_angles))
#     lowest_frame_landmarks = frame_landmarks[lowest_frame_idx]
#
#     # 특징 추출
#     features = []
#
#     # 오른쪽 무릎 각도
#     right_hip = lowest_frame_landmarks[24][:2]
#     right_knee = lowest_frame_landmarks[26][:2]
#     right_ankle = lowest_frame_landmarks[28][:2]
#     right_knee_angle = detector.calculate_angle(right_hip, right_knee, right_ankle)
#     features.append(right_knee_angle)
#
#     # 왼쪽 무릎 각도
#     left_hip = lowest_frame_landmarks[23][:2]
#     left_knee = lowest_frame_landmarks[25][:2]
#     left_ankle = lowest_frame_landmarks[27][:2]
#     left_knee_angle = detector.calculate_angle(left_hip, left_knee, left_ankle)
#     features.append(left_knee_angle)
#
#     # 오른쪽 힙 각도
#     right_shoulder = lowest_frame_landmarks[12][:2]
#     right_hip_angle = detector.calculate_angle(right_shoulder, right_hip, right_knee)
#     features.append(right_hip_angle)
#
#     # 왼쪽 힙 각도
#     left_shoulder = lowest_frame_landmarks[11][:2]
#     left_hip_angle = detector.calculate_angle(left_shoulder, left_hip, left_knee)
#     features.append(left_hip_angle)
#
#     features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
#
#     with torch.no_grad():
#         outputs = model(features_tensor.unsqueeze(0))
#         _, predicted = torch.max(outputs, 1)
#
#     return "Correct form" if predicted.item() == 1 else "Incorrect form"
#
#
# def main():
#     model_path = "lunge_classifier.pth"
#
#     video_paths = [
#         f"augmented_videos/lunge_test_correct{i}_augmented_{j}.mp4"
#         for i in range(1, 10)
#         for j in range(5)
#     ]
#
#     for video_path in video_paths:
#         try:
#             result = test_lunge_video(video_path, model_path)
#             print(f"Video {video_path}: {result}")
#         except Exception as e:
#             print(f"Error processing {video_path}: {e}")
#
#
# if __name__ == "__main__":
#     main()

#
# import os
# import torch
# import sys
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from lunge_anaylzer import LungePoseAnalyzer, LungeClassifier, LungeDataset
#
#
# def test_pose_analyzer():
#     print("Testing LungePoseAnalyzer...")
#
#     # Create an instance of LungePoseAnalyzer
#     analyzer = LungePoseAnalyzer()
#
#     # Test video paths (replace with your actual video paths)
#     test_videos = [
#         "augmented_videos/lunge_test_correct1_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct1_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct1_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct1_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct1_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct2_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct2_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct2_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct2_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct2_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct3_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct3_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct3_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct3_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct3_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct4_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct4_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct4_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct4_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct4_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct5_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct5_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct5_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct5_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct5_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct6_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct6_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct6_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct6_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct6_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct7_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct7_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct7_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct7_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct7_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct8_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct8_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct8_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct8_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct8_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct9_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct9_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct9_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct9_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct9_augmented_4.mp4",
#
#
#
#
#     ]
#
#     all_test_data = []
#     for video_path in test_videos:
#         if os.path.exists(video_path):
#             print(f"Processing video: {video_path}")
#             features = analyzer.extract_features(video_path)
#             all_test_data.extend(features)
#             print(f"Extracted {len(features)} frames")
#         else:
#             print(f"Warning: Video path not found - {video_path}")
#
#     # Print label counts
#     analyzer.print_label_counts()
#
#     return all_test_data
#
#
# def test_model_loading():
#     print("\nTesting Model Loading...")
#
#     # Create model instance
#     model = LungeClassifier()
#
#     # Try to load saved model
#     try:
#         model.load_state_dict(torch.load('lunge_classifier.pth'))
#         print("Model successfully loaded from lunge_classifier.pth")
#
#         # Basic inference test
#         test_input = torch.tensor([[90.0, 90.0, 45.0, 45.0]], dtype=torch.float32)
#         with torch.no_grad():
#             output = model(test_input)
#             predicted_class = torch.argmax(output, dim=1)
#             print("Test input prediction:", "Correct" if predicted_class.item() == 1 else "Incorrect")
#     except FileNotFoundError:
#         print("Error: Model file 'lunge_classifier.pth' not found. Run training first.")
#     except Exception as e:
#         print(f"An error occurred while loading the model: {e}")
#
#
# def test_dataset_creation(test_data):
#     print("\nTesting Dataset Creation...")
#
#     if not test_data:
#         print("No test data available")
#         return
#
#     try:
#         dataset = LungeDataset(test_data)
#
#         print(f"Dataset size: {len(dataset)}")
#         print("First data point:")
#         sample_input, sample_label = dataset[0]
#         print("Input shape:", sample_input.shape)
#         print("Label:", "Correct" if sample_label.item() == 1 else "Incorrect")
#     except Exception as e:
#         print(f"Error creating dataset: {e}")
#
#
# def main():
#     print("Starting Lunge Pose Analysis Test...")
#
#     # Test feature extraction
#     test_data = test_pose_analyzer()
#
#     # Test dataset creation
#     test_dataset_creation(test_data)
#
#     # Test model loading
#     test_model_loading()
#
#
# if __name__ == "__main__":
#     main()

# import os
# import cv2
# import torch
# import sys
# import mediapipe as mp
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from lunge_anaylzer import LungePoseAnalyzer, LungeClassifier, LungeDataset
#
#
# def verify_video_files(video_paths):
#     """
#     비디오 파일 검증 및 문제점 진단 함수
#     """
#     print("Video File Verification:")
#     valid_videos = []
#     for video_path in video_paths:
#         # 파일 존재 확인
#         if not os.path.exists(video_path):
#             print(f"❌ File Not Found: {video_path}")
#             continue
#
#         # 비디오 열기 시도
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             print(f"❌ Cannot Open Video: {video_path}")
#             cap.release()
#             continue
#
#         # 프레임 수 및 기본 정보 확인
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#         print(f"✅ Valid Video: {video_path}")
#         print(f"   Frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
#
#         valid_videos.append(video_path)
#         cap.release()
#
#     return valid_videos
#
#
# def test_pose_analyzer(video_paths):
#     print("\nTesting LungePoseAnalyzer...")
#
#     # Create an instance of LungePoseAnalyzer
#     analyzer = LungePoseAnalyzer()
#
#     all_test_data = []
#     total_extracted_frames = 0
#
#     for video_path in video_paths:
#         print(f"\nProcessing: {video_path}")
#
#         # 비디오 캡처
#         cap = cv2.VideoCapture(video_path)
#
#         # 비디오 프레임 수 출력
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         print(f"Total Video Frames: {total_frames}")
#
#         # 수동으로 프레임 처리
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             frame_count += 1
#
#             # 디버그: 첫 10프레임만 처리
#             if frame_count > 10:
#                 break
#
#         cap.release()
#
#         # 수동 특징 추출
#         features = analyzer.extract_features(video_path)
#
#         if features:
#             print(f"Successfully extracted {len(features)} frames")
#             total_extracted_frames += len(features)
#             all_test_data.extend(features)
#         else:
#             print(f"❌ No frames extracted from {video_path}")
#
#     print(f"\nTotal Extracted Frames: {total_extracted_frames}")
#     analyzer.print_label_counts()
#
#     return all_test_data
#
#
# def test_model_inference(test_data):
#     print("\nTesting Model Inference...")
#
#     # 모델 로드
#     model = LungeClassifier()
#     model.load_state_dict(torch.load('lunge_classifier.pth'))
#     model.eval()
#
#     # 데이터셋 생성
#     dataset = LungeDataset(test_data)
#
#     # 전체 데이터에 대한 예측
#     correct_predictions = 0
#     total_predictions = 0
#
#     with torch.no_grad():
#         for inputs, labels in torch.utils.data.DataLoader(dataset, batch_size=32):
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#
#             total_predictions += labels.size(0)
#             correct_predictions += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct_predictions / total_predictions
#     print(f"Model Accuracy: {accuracy:.2f}%")
#     print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
#
#
# def main():
#     # 테스트 비디오 경로 목록
#     test_videos = [
#         "augmented_videos/lunge_test_correct1_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct1_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct1_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct1_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct1_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct2_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct2_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct2_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct2_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct2_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct3_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct3_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct3_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct3_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct3_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct4_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct4_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct4_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct4_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct4_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct5_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct5_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct5_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct5_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct5_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct6_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct6_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct6_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct6_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct6_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct7_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct7_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct7_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct7_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct7_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct8_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct8_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct8_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct8_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct8_augmented_4.mp4",
#         "augmented_videos/lunge_test_correct9_augmented_0.mp4",
#         "augmented_videos/lunge_test_correct9_augmented_1.mp4",
#         "augmented_videos/lunge_test_correct9_augmented_2.mp4",
#         "augmented_videos/lunge_test_correct9_augmented_3.mp4",
#         "augmented_videos/lunge_test_correct9_augmented_4.mp4",
#     ]
#
#     # 비디오 파일 검증
#     valid_videos = verify_video_files(test_videos)
#
#     if not valid_videos:
#         print("No valid videos found. Cannot proceed.")
#         return
#
#     # 특징 추출
#     test_data = test_pose_analyzer(valid_videos)
#
#     if test_data:
#         # 모델 추론 테스트
#         test_model_inference(test_data)
#     else:
#         print("No test data available for inference.")
#
#
# if __name__ == "__main__":
#     main()

import os
import cv2
import torch
import mediapipe as mp
import numpy as np
from torch.utils.data import Dataset, DataLoader


class LungePoseAnalyzer:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

    def find_lowest_point_frame(self, video_path):
        video = cv2.VideoCapture(video_path)
        angles_by_frame = []

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
            return None

        avg_angles = [sum(frame.values()) / len(frame) for frame in angles_by_frame]
        lowest_frame_idx = avg_angles.index(min(avg_angles))

        return {
            'frame_index': lowest_frame_idx,
            'angles': angles_by_frame[lowest_frame_idx]
        }


class LungeClassifier(torch.nn.Module):
    def __init__(self):
        super(LungeClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def classify_lowest_frames(video_paths):
    # 모델 로드
    model = LungeClassifier()
    model.load_state_dict(torch.load('lunge_classifier.pth'))
    model.eval()

    # 분석 결과 저장
    analysis_results = []

    # 각 비디오에 대해 처리
    for video_path in video_paths:
        # 최저점 프레임 찾기
        analyzer = LungePoseAnalyzer()
        lowest_point = analyzer.find_lowest_point_frame(video_path)

        if lowest_point is None:
            print(f"No landmarks found in {video_path}")
            continue

        # 모델 입력 준비
        input_tensor = torch.tensor([
            [
                lowest_point['angles']['right_knee'],
                lowest_point['angles']['left_knee'],
                lowest_point['angles']['right_hip'],
                lowest_point['angles']['left_hip']
            ]
        ], dtype=torch.float32)

        # 추론
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()

        analysis_results.append({
            'video_path': video_path,
            'frame_index': lowest_point['frame_index'],
            'angles': lowest_point['angles'],
            'prediction': 'Correct' if predicted_class == 1 else 'Incorrect',
            'confidence': confidence
        })

    return analysis_results


def main():
    test_videos = [
        "data/lunge_test_correct1.mp4",
        "data/lunge_test_correct2.mp4",
        "data/lunge_test_correct3.mp4",
        "data/lunge_test_correct4.mp4",
        "data/lunge_test_correct5.mp4",
        "data/lunge_test_correct6.mp4",
        "data/lunge_test_correct7.mp4",
        "data/lunge_test_correct8.mp4",
        "data/lunge_test_correct9.mp4",




    ]

    results = classify_lowest_frames(test_videos)

    # 결과 출력
    print("\n분석 결과:")
    for result in results:
        print(f"\n비디오: {result['video_path']}")
        print(f"최저점 프레임 인덱스: {result['frame_index']}")
        print(f"각도: {result['angles']}")
        print(f"예측: {result['prediction']} (신뢰도: {result['confidence']:.2%})")


if __name__ == "__main__":
    main()