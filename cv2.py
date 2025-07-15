import cv2
import numpy as np
import serial
import time
import queue
import threading

DEBUG = True
ROI_HEIGHT = 240
MIN_AREA = 300
ALPHA = 0.2
ANGLE_SMOOTHING = 0.2
UART_PORT = '/dev/ttyTHS1'
UART_BAUDRATE = 9600

uart_queue = queue.Queue()

def uart_sender():
    ser = serial.Serial(UART_PORT, UART_BAUDRATE)
    time.sleep(2)
    while True:
        try:
            command = uart_queue.get(timeout=1)
            ser.write((command + "\n").encode())
            uart_queue.task_done()
        except queue.Empty:
            pass
        time.sleep(0.001)

threading.Thread(target=uart_sender, daemon=True).start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    raise SystemError("CUDA destekli GPU bulunamadı.")
print("✅ CUDA kablo takibi başlatıldı.")

stream = cv2.cuda_Stream()
gpu_close = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1, np.ones((5, 5), np.uint8))

gpu_frame = cv2.cuda_GpuMat()
gpu_hsv = cv2.cuda_GpuMat()
gpu_mask = cv2.cuda_GpuMat()

cx_ema = None
angle_ema = None
son_komut = "F300"

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    gpu_frame.upload(frame, stream)
    gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV, stream)

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 50], dtype=np.uint8)
    gpu_mask = cv2.cuda.inRange(gpu_hsv, lower_black, upper_black)

    gpu_mask = gpu_close.apply(gpu_mask)
    mask = gpu_mask.download(stream)
    stream.waitForCompletion()

    roi = mask[-ROI_HEIGHT:, :]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    command = son_komut
    position_text = "Kablo yok - 90° sağa dön"
    kablo_var = False

    if contours:
        valid = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
        if valid:
            kablo_var = True
            largest = max(valid, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                [vx, vy, x, y] = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = np.degrees(np.arctan2(vy, vx))

                cx_ema = cx if cx_ema is None else int(ALPHA * cx + (1 - ALPHA) * cx_ema)
                angle_ema = angle if angle_ema is None else ANGLE_SMOOTHING * angle + (1 - ANGLE_SMOOTHING) * angle_ema

                center = frame.shape[1] // 2
                margin = 40

                if center - margin < cx_ema < center + margin:
                    command = "F300"
                    position_text = f"Kablo ortada ({angle_ema:.1f}°)"
                elif cx_ema >= center + margin:
                    command = "R300"
                    position_text = f"Kablo sağda ({angle_ema:.1f}°)"
                elif cx_ema < center - margin:
                    if angle_ema > 20:
                        command = "R300"
                        position_text = f"Kablo solda ama sağa kıvrılıyor ({angle_ema:.1f}°)"
                    else:
                        command = "L300"
                        position_text = f"Kablo solda ({angle_ema:.1f}°)"

    if not kablo_var:
        command = "Q300"  # ⬅️ DÜZELTİLEN KISIM: Sağa keskin dönüş

    uart_queue.put(command)
    son_komut = command

    if DEBUG:
        cv2.putText(frame, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if kablo_var:
            cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy + frame.shape[0] - ROI_HEIGHT), 5, (0, 0, 255), -1)
            cv2.line(frame, (center, 0), (center, frame.shape[0]), (255, 255, 255), 1)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Kablo Takibi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
uart_queue.join()



