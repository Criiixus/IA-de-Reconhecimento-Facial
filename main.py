import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Usa a câmera do dispositivo (DroidCam costuma ser o índice 1 ou 2, teste se for o caso)
cap = cv2.VideoCapture("http://192.168.0.156:4747/video")


with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("❌ Erro ao capturar frame")
            break

        # Conversão BGR → RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        # Volta pra BGR pra exibir
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.detections:
            print("✅ Rosto detectado!")
            for detection in results.detections:
                mp_drawing.draw_detection(image_bgr, detection)
        else:
            print("🔍 Nenhum rosto detectado")

        cv2.imshow('Reconhecimento Facial', image_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
