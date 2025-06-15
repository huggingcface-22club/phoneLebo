# import cv2
# import mediapipe as mp
# import math

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

# def find_distance(p1, p2):
#     return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# while True:
#     success, img = cap.read()
#     if not success:
#         break

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(img_rgb)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             ih, iw = img.shape[:2]
#             lm = face_landmarks.landmark

#             # 예시: 왼쪽 눈 중심 (33), 오른쪽 눈 중심 (263)
#             p1 = (int(lm[33].x * iw), int(lm[33].y * ih))
#             p2 = (int(lm[263].x * iw), int(lm[263].y * ih))

#             # 거리 계산
#             d = find_distance(p1, p2)

#             # 시각화
#             cv2.circle(img, p1, 5, (0, 255, 0), -1)
#             cv2.circle(img, p2, 5, (0, 255, 0), -1)
#             cv2.line(img, p1, p2, (255, 0, 0), 2)
#             cv2.putText(img, f"Eye Dist: {int(d)} px", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

#             # 얼굴 전체 랜드마크도 그리기
#             mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

#     cv2.imshow("Face Distance Tracker", img)
#     if cv2.waitKey(1) == 27:  # ESC 키로 종료
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import math

# 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

def find_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

last_angles = [0, 0, 0]  # 예시용 Roll 각도 (0으로 고정)

while True:
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    h, w = image.shape[:2]
    face_detected = False

    if results.multi_face_landmarks:
        face_detected = True
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 메시 스타일 적용
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )

            # 예시용 Roll 각도 출력
            # cv2.putText(image, f'Roll: {last_angles[2]:.2f}', (20, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            # 주요 포인트 수집
            face_img = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 9, 57, 130, 287, 359]:  # 예: 코끝, 입, 눈꼬리 등
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_img.append((x, y))
                    #cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

            # 거리 예시: 코끝(1)과 턱(152) 거리 측정
            nose = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[152]
            p1 = int(nose.x * w), int(nose.y * h)
            p2 = int(chin.x * w), int(chin.y * h)

            distance = find_distance(p1, p2)
            #cv2.line(image, p1, p2, (0, 0, 255), 2)
            cv2.putText(image, f'Dist: {int(distance)} px', (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Face Mesh Distance", image)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
