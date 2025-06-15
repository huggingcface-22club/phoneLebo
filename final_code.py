import cv2
import math
import time
import threading
import math
import numpy as np
import mediapipe as mp
from lerobot.common.motors.motors_bus import Motor, MotorNormMode, MotorCalibration
from lerobot.common.motors.feetech.feetech import FeetechMotorsBus

# === 전역 변수 ===
last_angles = [0.0, 0.0, 0.0]
face_detected = False
face_center = (0, 0)
angle_history = {'pitch': [], 'yaw': [], 'roll': []}
MAX_HISTORY = 5
MAX_ANGLE_DELTA = 15
frame_width = 0
frame_height = 0
last_face_position_time = None
last_face_position = None
terminate_signal = False

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(2)

# 거리 계산 함수
def find_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# 회전 행렬을 각도로 변환하는 함수
def rotation_matrix_to_angles(rotation_matrix):
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi

# 각도 부드럽게 처리 함수
def smooth_angle(name, new_value, angle_history, MAX_ANGLE_DELTA=15, MAX_HISTORY=5):
    history = angle_history.get(name, [])
    if history and abs(new_value - history[-1]) > MAX_ANGLE_DELTA:
        new_value = history[-1]
    history.append(new_value)
    if len(history) > MAX_HISTORY:
        history.pop(0)
    angle_history[name] = history
    return sum(history) / len(history)

def face_tracking_loop():
    global last_angles, face_detected, face_center, frame_width, frame_height, last_face_position_time, last_face_position, terminate_signal
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        h, w, _= image.shape
        frame_width, frame_height = w, h
        #face_detected = False

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
                # 회전 각도 출력
                cv2.putText(image, f'Roll: {last_angles[2]:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                
                # 주요 포인트 수집
                face_img = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:  # 예: 코끝, 입, 눈꼬리 등
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_img.append((x, y))

                # 거리 예시: 코끝(1)과 턱(152) 거리 측정
                nose = face_landmarks.landmark[1]
                chin = face_landmarks.landmark[152]
                p1 = int(nose.x * w), int(nose.y * h)
                p2 = int(chin.x * w), int(chin.y * h)

                distance = find_distance(p1, p2)
                cv2.putText(image, f'Dist: {int(distance)} px', (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # 얼굴의 회전 각도 계산
                face_real = np.array([
                    [285, 528, 200], [285, 371, 152], [197, 574, 128],
                    [173, 425, 108], [360, 574, 128], [391, 425, 108]
                ], dtype=np.float64)

                if len(face_img) == 6:
                    face_img = np.array(face_img, dtype=np.float64)
                    cam_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    success, rvec, tvec = cv2.solvePnP(face_real, face_img, cam_matrix, dist_matrix)
                    rot_matrix, _ = cv2.Rodrigues(rvec)
                    angles = rotation_matrix_to_angles(rot_matrix)
                    last_angles = [
                        smooth_angle('pitch', float(angles[0]), angle_history),
                        smooth_angle('yaw', float(angles[1]), angle_history),
                        smooth_angle('roll', float(angles[2]), angle_history)
                    ]

                    # 얼굴 위치에 따른 화면 위치 출력
                    face_center = tuple(np.mean(face_img[:2], axis=0))
                    fc_x, fc_y = face_center
                    vert, horiz = '', ''
                    if fc_x < w / 3:
                        horiz = 'LEFT'
                    elif fc_x > 2 * w / 3:
                        horiz = 'RIGHT'
                    else:
                        horiz = 'CENTER'

                    if fc_y < h / 3:
                        vert = 'UP'
                    elif fc_y > 2 * h / 3:
                        vert = 'DOWN'
                    else:
                        vert = 'CENTER'

                    grid_pos = f'{vert}-{horiz}'
                    cv2.putText(image, f'POS: {grid_pos}', (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                
        else:
            face_detected = False

        cv2.imshow("Face Mesh and Distance", image)
        if cv2.waitKey(5) & 0xFF == 27:
            terminate_signal = True
            break

        # cap.release()
        # cv2.destroyAllWindows()

def get_pitch_yaw_roll():
    return last_angles

threading.Thread(target=face_tracking_loop, daemon=True).start()

motors = {
    "shoulder_yaw": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.DEGREES),
    "shoulder_pitch": Motor(id=2, model="sts3215", norm_mode=MotorNormMode.DEGREES),
    "elbow_pitch": Motor(id=3, model="sts3215", norm_mode=MotorNormMode.DEGREES),
    "wrist_pitch": Motor(id=4, model="sts3215", norm_mode=MotorNormMode.DEGREES),
    "wrist_roll": Motor(id=5, model="sts3215", norm_mode=MotorNormMode.DEGREES),
    "gripper": Motor(id=6, model="sts3215", norm_mode=MotorNormMode.DEGREES),
}

calibration = {
    name: MotorCalibration(id=motor.id, drive_mode=0, homing_offset=2048, range_min=0, range_max=4095)
    for name, motor in motors.items()
}

bus = FeetechMotorsBus(port="/dev/ttyACM0", motors=motors, calibration=calibration)
print("Connecting to robot...")
if not bus.is_connected:
    bus.connect()
try:
    bus.enable_torque()
except ConnectionError as e:
    print(f"일부 모터에 토크 설정 실패: {e}")

base_angles = {
    "shoulder_yaw": 0,
    "shoulder_pitch": 45,
    "elbow_pitch": -45,
    "wrist_pitch": 105,
    "wrist_roll" : 15,
}

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def move_motor_smooth(name, target_angle, steps=30, max_speed=2500):
    current = bus.read("Present_Position", name, normalize=True)
    delta = target_angle - current
    for i in range(1, steps + 1):
        progress = i / steps
        interp = current + delta * progress
        factor = math.sin(progress * math.pi)
        speed = max(30, int(max_speed * factor))
        bus.write("Goal_Velocity", name, speed, normalize=False)
        bus.write("Goal_Position", name, interp, normalize=True)
        time.sleep(0.02)

def move_motor_smooth_init(name, target, speed=5):
    try:
        current = bus.read("Present_Position", name, normalize=True)
    except RuntimeError as e:
        print(f"[경고] '{name}' 현재 위치 읽기 실패 → 해당 관절 건너뜀\n→ {e}")
        return

    steps = int(abs(target - current) / speed) + 1
    for i in range(steps):
        interp = current + (target - current) * (i + 1) / steps
        try:
            bus.write("Goal_Position", name, interp, normalize=True)
        except RuntimeError as e:
            print(f"[경고] '{name}' 이동 중 실패 → 중단\n→ {e}")
            break
        time.sleep(0.01)


print("Tracking face angles... (Ctrl+C to exit)")
lost_face_start = None
face_lost_threshold = 3.0

## gripper 설정
def close_gripper_with_lock(threshold=0.5, max_wait=3.0):
    print("그리퍼 닫기 및 위치 고정 시도")
    try:
        start_time = time.time()
        prev_pos = bus.read("Present_Position", "gripper", normalize=True)
        move_motor_smooth("gripper", 0)

        while True:
            time.sleep(0.1)
            current_pos = bus.read("Present_Position", "gripper", normalize=True)
            delta = abs(current_pos - prev_pos)

            # 변화가 거의 없으면 물체가 끼인 것으로 간주
            if delta < threshold:
                print(f"위치 고정: {current_pos:.2f} (변화량: {delta:.2f})")
                break

            prev_pos = current_pos

            if time.time() - start_time > max_wait:
                print("timeout: 물체 없음 또는 정상 닫힘")
                break

        # 모터를 현재 위치에 고정 (토크 유지 상태)
        bus.write("Goal_Position", "gripper", current_pos, normalize=True)

    except RuntimeError as e:
        print(f"⚠ 그리퍼 동작 실패: {e}")
def open_gripper_if_safe(threshold=80):
    try:
        grip_pos = bus.read("Present_Position", "gripper", normalize=True)
        print(f"현재 gripper 위치: {grip_pos:.2f}")
    except RuntimeError as e:
        print(f"gripper 위치 읽기 실패 → 열기 생략\n→ {e}")
        return

    if grip_pos > threshold:
        print("이미 열려 있음 → 생략")
        return

    try:
        print("gripper 열기 시도")
        move_motor_smooth("gripper", 100)
    except RuntimeError as e:
        print(f"⚠ gripper 열기 실패 → 과부하 가능성 있음\n→ {e}")


## main Thread
try:
    # 위치 초기화 
    for name, angle in base_angles.items():
        move_motor_smooth(name, angle)
    # 얼굴이 사라진 후, 그리퍼 열기 및 닫기 동작 예시
    open_gripper_if_safe()
    time.sleep(3.0)
    close_gripper_with_lock()  # 자동 고정 로직 포함
    while True:
        if terminate_signal:
            raise KeyboardInterrupt

        if not face_detected:
            if lost_face_start is None:
                lost_face_start = time.time()
            elif time.time() - lost_face_start > face_lost_threshold:
                print("얼굴 사라짐 → 이전 위치 또는 기본자세")
                for name, angle in base_angles.items():
                    move_motor_smooth_init(name, angle)
                if last_face_position_time and time.time() - last_face_position_time < 3:
                    print("↩ 직전 위치로 복귀")
                    face_center = last_face_position
                else:
                    print("기본 자세 복귀 시도")
                    for name, angle in base_angles.items():
                        move_motor_smooth_init(name, angle)
                lost_face_start = None
            time.sleep(0.1)
            continue
        else:
            lost_face_start = None

        r = get_pitch_yaw_roll()
        roll_angle = r[2]

        if -10 <= roll_angle <= 10:
            print(f"Roll 정면 유지: {roll_angle:.2f}° → 모터 정지")
            delta = clamp(0, 0, 0)
        else:
            delta = clamp(roll_angle, -40, 40)
        base_angles["wrist_roll"] = clamp(delta, -50, 50)
        print(f"Roll 제어: {roll_angle:.2f}° → wrist_roll {base_angles['wrist_roll']:.2f}°")
        move_motor_smooth("wrist_roll", base_angles["wrist_roll"])

        base_angles["shoulder_pitch"] = clamp(50, 0, 120)
        base_angles["elbow_pitch"] = clamp(-60, -45, -120)

        x, y = face_center
        in_center_x = frame_width / 3 < x < 2 * frame_width / 3
        in_center_y = frame_height / 3 < y < 2 * frame_height / 3

        if in_center_x and in_center_y:
            print("얼굴이 중앙에 위치함 → 정지")
            continue

        if not in_center_x:
            while not (frame_width / 3 < x < 2 * frame_width / 3):
                if x < frame_width / 3:
                    base_angles["shoulder_yaw"] = clamp(base_angles["shoulder_yaw"] + 3, -90, 90)
                elif x > 2 * frame_width / 3:
                    base_angles["shoulder_yaw"] = clamp(base_angles["shoulder_yaw"] - 3, -90, 90)
                move_motor_smooth("shoulder_yaw", base_angles["shoulder_yaw"])
                time.sleep(0.05)
                x, _ = face_center

        if not in_center_y:
            while not (frame_height / 3 < y < 2 * frame_height / 3):
                if y < frame_height / 3:
                    base_angles["shoulder_pitch"] = clamp(base_angles["shoulder_pitch"] - 3, 0, 90)
                elif y > 2 * frame_height / 3:
                    base_angles["shoulder_pitch"] = clamp(base_angles["shoulder_pitch"] + 3, 0, 90)
                move_motor_smooth("shoulder_pitch", base_angles["shoulder_pitch"])
                time.sleep(0.05)
                _, y = face_center

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n[사용자 종료] 프로그램 종료: 로봇팔 초기 위치로 복귀 중...")
    all_returned = True
    for name, angle in base_angles.items():
        move_motor_smooth(name, angle)
        current = bus.read("Present_Position", name, normalize=True)
        if abs(current - angle) > 1:
            all_returned = False

    if all_returned:
        cap.release()                      # 카메라 해제
        cv2.destroyAllWindows()
        open_gripper_if_safe()
        bus.disable_torque()
        bus.disconnect()
        print("종료 완료")
    else:
        open_gripper_if_safe()
        print("일부 모터가 초기 위치에 도달하지 못함")

