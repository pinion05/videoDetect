import tkinter as tk  
from tkinter import filedialog  
import cv2  
from PIL import Image, ImageTk  
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import time
import threading  
import csv
defaultCuttingTime = 1
# ResNet50 모델을 로드합니다.
model = ResNet50()
# 결과를 기록할 텍스트 파일의 이름을 전역 변수로 선언
result_file_name = ""

#이미지를 전처리하는 함수
def upload_image(image, video_time):
    global result_file_name  # 전역 변수 사용
    image_array = img_to_array(image)
    image_array = tf.keras.preprocessing.image.smart_resize(image_array, (224, 224))
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]  # 상위 1개만 가져옴
    imagenetID, label, prob = decoded_predictions[0]  # 결과 unpacking
    result_text = f"1. {label}: {prob * 100:.2f}%\n"  # 결과 문자열 생성
    # 결과를 Label에 업데이트
    result_label.config(text=result_text)
    # 현재 영상 재생 시간과 결과를 CSV 형식으로 기록
    with open(result_file_name, "a", encoding='utf-8', newline='') as file:  # UTF-8 인코딩으로 파일 열기
        writer = csv.writer(file)
        writer.writerow([video_time, f"{label}: {prob * 100:.2f}%"])  # 영상 재생 시간과 상위 1개 결과 기록
def getNull(val, val2):
    if val == "":
        return val2
    return val
class VideoPlayer:
    def __init__(self, root):
        self.root = root  
        self.root.title("Video Player")  
        # 비디오 표시를 위한 Label 위젯 생성
        self.video_label = tk.Label(root, padx=5)  
        self.video_label.pack()  
        # 커팅타입 사용자 설정을 위한 entry
        self.cuttingTimeInput = tk.Entry(root)
        self.cuttingTimeInput.pack()
        self.cuttingTimeInput.bind("<Return>", self.update_frame_interval)  # Enter 키 입력 시 업데이트
        # 예측 결과를 표시하기 위한 Label
        global result_label
        result_label = tk.Label(root, text="", justify="left")
        result_label.pack(pady=10)
        # 비디오 파일 선택 버튼 생성
        self.upload_btn = tk.Button(root, text="Upload", command=self.upload_video)  
        self.upload_btn.pack()  
        self.cap = None  
        self.last_frame_time = time.time()  
        self.frame_interval = defaultCuttingTime  # 초기 프레임 간격 설정
    def update_frame_interval(self, event):
        try:
            # Entry에서 값을 읽어 실수로 변환
            self.frame_interval = float(self.cuttingTimeInput.get())
            print(f"프레임 간격이 {self.frame_interval}초로 설정되었습니다.")
        except ValueError:
            print("유효한 정수를 입력하세요.")
    def upload_video(self):
        global result_file_name  # 전역 변수 사용
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])  
        if file_path:  
            self.cap = cv2.VideoCapture(file_path)  
            # 같은 이름의 txt 파일 생성 (확장자 변경)
            result_file_name = file_path.rsplit('.', 1)[0] + ".csv"  # .txt 대신 .csv로 변경
            with open(result_file_name, "w", encoding='utf-8', newline='') as file:  # UTF-8 인코딩으로 파일 초기화
                writer = csv.writer(file)
                writer.writerow(["영상재생시간", "상위확률"])  # CSV 헤더 기록
            self.play_video()  
    def play_video(self):
        if self.cap.isOpened():  
            ret, frame = self.cap.read()  
            if ret:  
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                img = Image.fromarray(frame)  
                current_time = time.time()
                video_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 영상 재생 시간 (초 단위)
                if current_time - self.last_frame_time >= self.frame_interval:
                    self.last_frame_time = current_time  
                    threading.Thread(target=upload_image, args=(img, video_time)).start()  
                img = ImageTk.PhotoImage(img)  
                self.video_label.img = img  
                self.video_label.config(image=img)  
                self.video_label.after(33, self.play_video)  
            else:  
                self.cap.release()  
# tkinter GUI 설정
root = tk.Tk()  
video_player = VideoPlayer(root)  
root.mainloop()  
