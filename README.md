# HED-3D
HED-3D is a real-time monocular relative-depth model for refueling localization. It detects hierarchical ellipses to build a structured ROI/mask, generates robust teacher pseudo-labels, and distills a lightweight MobileNetV3 student with uncertainty heads and SoftGate fusion for accurate, low-jitter depth on edge devices.

HED process：
![第三章](https://github.com/user-attachments/assets/168cb49b-2063-4b61-a865-4a03834c45b5)

HED effect：

<img width="700" height="933" alt="1563" src="https://github.com/user-attachments/assets/b1c48e17-9fdb-418b-a263-ffb9b3abe674" />
<img width="651" height="756" alt="2238" src="https://github.com/user-attachments/assets/b05adf3d-9f01-4728-80ae-a0ce500d250f" />
<img width="810" height="1440" alt="spout_2" src="https://github.com/user-attachments/assets/509a4643-4080-4e65-b336-456112890304" />

Network architecture：
![网络架构](https://github.com/user-attachments/assets/7ef74dc0-a95b-4f03-862b-4b3eab80c161)

Distillation effect：
![1368_oilmouth_1_compare](https://github.com/user-attachments/assets/9af9a2c3-bd6d-4262-9b7d-7e40501de64a)
![1364_oilmouth_1_compare](https://github.com/user-attachments/assets/b0195f23-df64-4d8c-9ac3-c3c3be8530e4)
![1363_oilmouth_1_compare](https://github.com/user-attachments/assets/82765c92-50f8-4125-87b3-41312d46d939)
