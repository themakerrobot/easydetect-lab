Pibo로 보내기 — easydetect lab에서 학습한 모델 (작업 #{JOB})
==========================================================

들어 있는 것
  model.onnx   학습한 모델 (입력 {IMGSZ}x{IMGSZ}, 쿼리 {QUERIES}개)
  labels.txt   클래스 이름, 한 줄에 하나: {CLASSES}
  detect.py    모델을 돌리는 프로그램 (numpy + opencv + onnxruntime만 있으면 돼요)
  README.txt   이 파일

Pibo에 필요한 것
  Python 3, numpy, opencv(cv2), onnxruntime 1.13 이상.
  easydetect나 torch는 필요 없어요. 있는지 확인:
      python3 -c "import numpy, cv2, onnxruntime; print(onnxruntime.__version__)"
  onnxruntime이 1.13보다 낮으면(모델이 opset 17을 써요):
      pip3 install -U onnxruntime
  64비트 OS(aarch64)에서는 pip로 바로 깔려요. 32비트 OS(armv7l)에는 공식
  onnxruntime 패키지가 없어요 — `uname -m`으로 확인하세요.

Pibo로 옮기기
  1. 이 zip을 풀어서 폴더째 Pibo로 복사해요. 예 (같은 네트워크일 때):
         scp -r pibo-job{JOB} pi@<Pibo의 IP>:~/
     USB 메모리로 옮겨도 돼요.
  2. Pibo의 터미널(SSH)에서:
         cd ~/pibo-job{JOB}
         python3 detect.py 사진.jpg                 # 사진 한 장: 찾은 것을 글로 보여 줘요
         python3 detect.py 사진.jpg --save out.jpg  # 상자를 그린 사진도 저장
         python3 detect.py 0                        # 카메라 0번, Ctrl+C로 멈춰요
         python3 detect.py 0 --show                 # 화면이 있으면 창으로 보여 줘요
     --conf 0.6처럼 확신 기준을 바꿀 수 있어요 (기본 0.5, lab과 같아요).

내 로봇 코드에서 쓰기
  detect.py가 있는 폴더에서:
      from detect import detect
      for x1, y1, x2, y2, score, name in detect(frame):   # frame: OpenCV BGR 사진
          print(name, round(score, 2))
  사진 경로를 넘겨도 돼요: detect("사진.jpg"). 결과는 확신이 높은 것부터,
  사진의 픽셀 좌표예요. 처음 부를 때 모델을 읽어요(1~2초).

알아 두기
  - lab의 추론 화면과 같은 방법으로 계산해서, 같은 사진이면 같은 상자와 점수가 나와요.
  - OpenCV는 한글을 그리지 못해서, 그림 위의 한글 클래스 이름은 번호(#0, #1…)로 보여요.
    글로 출력되는 이름은 한글 그대로예요.
  - 라즈베리파이 4에서는 사진 한 장에 대략 0.1~0.3초가 걸릴 거예요 (측정 전 추정).

openpibo에서 쓰기 (사물 인식을 이 모델로 바꾸기)
  openpibo의 Detect는 사물 인식기를 detect.object_detector에 둬요. 이 폴더의
  Detector가 YoloOnnx와 같은 방식으로 불리니, 한 줄로 바꿔 끼우면 돼요:

      import sys; sys.path.insert(0, "/home/pi/pibo-job{JOB}")
      from detect import Detector
      from openpibo.vision_detect import Detect

      detect = Detect()
      detect.object_detector = Detector("/home/pi/pibo-job{JOB}/model.onnx")
      items = detect.detect_object(img)        # 이름·점수·상자 목록
      detect.detect_object_vis(img, items)     # 한글 이름도 그려져요 (PIL)

  원래 모델(yolo11s)로 돌아가려면 detect.load_object_model()을 부르면 돼요.
