# rov_cable_yolo



# 🔍 Kablo Takip Sistemi — YOLO + CUDA (Jetson Xavier Uyumlu)

Bu proje, su altı görevlerinde siyah kabloyu tespit etmek ve konumuna göre yön tayini yapmak amacıyla geliştirilmiştir. YOLOv8/YOLOv12 ile eğitilmiş bir model kullanılarak kablo gerçek zamanlı olarak tespit edilir. OpenCV’nin `cv2.cuda` modülüyle Jetson Xavier cihazlarında GPU hızlandırması sağlanır.

---

## 🚀 Özellikler

- 🎯 YOLO tabanlı kablo tespiti (eğitilmiş `best.pt` modeli ile)
- ⚡ CUDA destekli OpenCV kullanımı (GPU hızlandırmalı görüntü işleme)
- 🎮 Komut üretimi: `STRAFE_LEFT`, `FORWARD`, `STRAFE_RIGHT`
- 💻 Sadece terminal ile çalıştırılabilir, IDE gerekmez
- 🧠 Xavier NX üzerinde optimize edilmiş

---

## 📁 Proje Yapısı

