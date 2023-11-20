import cv2
import numpy as np
import mediapipe as mp
from scipy.integrate import quad
from math import pi
    
# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load nama kelas (coco.names)
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load model Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

# Jarak kita terhadap kamera (dalam meter)
jarak_kamera = 300

# Inisialisasi variabel untuk menyimpan posisi y tulisan
text_y_positions = {
    "Estimated Height": 20,
    "Shoulder Distance": 40,
    "Shoulder Radius": 60,
    "Height Measure": 80,
    "Estimated Weight": 100,
}

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()

    # Deteksi objek menggunakan YOLO
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Informasi deteksi objek
    person_heights = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class 0 represents 'person'
                w = detection[2] * width
                h = detection[3] * height

                # Hitung tinggi berdasarkan perbandingan tinggi sebenarnya dengan tinggi di dalam layar
                person_height = (h / height) * jarak_kamera
                person_heights.append((person_height, detection))

    # Pilih deteksi yang paling dekat dengan tengah layar
    if person_heights:
        person_heights.sort(key=lambda x: abs(x[1][0] - 0.5))
        selected_height, selected_detection = person_heights[0]

        x = int(selected_detection[0] * width - w / 2)
        y = int(selected_detection[1] * height - h / 2)

        # Gambar kotak di sekitar objek
        cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Estimated Height: {selected_height:.2f} cm",
            (10, text_y_positions["Estimated Height"]),  # Updated coordinates for top-left corner
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        # Deteksi pose menggunakan Mediapipe Pose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Gambar titik-titik pose pada frame
        if results.pose_landmarks:
            # Gambar titik bahu kiri
            shoulder_left = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
            cv2.circle(frame, (int(shoulder_left[0]), int(shoulder_left[1])), 5, (255, 0, 0), cv2.FILLED)

            # Gambar titik bahu kanan
            shoulder_right = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
            cv2.circle(frame, (int(shoulder_right[0]), int(shoulder_right[1])), 5, (255, 0, 0), cv2.FILLED)

            # Gambar titik ujung kaki kanan
            ankle_right = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height)
            cv2.circle(frame, (int(ankle_right[0]), int(ankle_right[1])), 5, (255, 0, 0), cv2.FILLED)

            # Pengukuran lebar bahu kiri ke kanan (setengah dari ukuran aslinya)
            shoulder_distance_cm = np.sqrt((shoulder_right[0] - shoulder_left[0])**2 + (shoulder_right[1] - shoulder_left[1])**2) / width * jarak_kamera 

            # Pengukuran tinggi dari bahu kanan ke ujung kaki kanan
            height_measure_cm = np.abs(ankle_right[1] - shoulder_right[1]) / height * jarak_kamera

            # Radius bahu (setengah dari panjang asli lebar bahu)
            shoulder_radius_cm = shoulder_distance_cm / 2

            # Hitung volume tabung
            def integrand(y):
                r_squared = (shoulder_radius_cm) ** 2  # Convert cm to meters
                return pi * r_squared

            volume, _ = quad(integrand, 0, height_measure_cm)  # Convert cm to meters

            # Estimasi berat badan (density manusia ~1 kg/L)
            estimated_weight_kg = volume   # Convert to grams
            estimated_weight_kg /= 1000  # Convert grams to kilograms

            # Tampilkan hasil pengukuran
            for measurement, value in zip(["Shoulder Distance", "Shoulder Radius", "Height Measure", "Estimated Weight"], [shoulder_distance_cm, shoulder_radius_cm, height_measure_cm, estimated_weight_kg]):
                cv2.putText(
                    frame,
                    f"{measurement}: {value:.2f} {'cm' if measurement in ['Shoulder Distance', 'Shoulder Radius'] else 'kg'}",
                    (10, text_y_positions[measurement]),  # Updated coordinates for top-left corner
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    # Tampilkan frame
    cv2.imshow("Webcam", frame)

    # Hentikan program dengan menekan tombol 'q'
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord(" "):  # Tombol spasi ditekan
        # Buat jendela baru untuk menampilkan berat badan dan tinggi badan
        result_window = np.ones((200, 400, 3), dtype=np.uint8) * 255  # Jendela berukuran 400x200 pixel
        result_text_y = 20

        # Tampilkan hasil pengukuran pada jendela baru
        for measurement, value in zip(["Estimated Height", "Estimated Weight"], [selected_height, estimated_weight_kg]):
            cv2.putText(
                result_window,
                f"{measurement}: {value:.2f} {'cm' if measurement == 'Estimated Height' else 'kg'}",
                (10, result_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
            result_text_y += 40

        # Tampilkan jendela baru
        cv2.imshow("Results", result_window)

# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()