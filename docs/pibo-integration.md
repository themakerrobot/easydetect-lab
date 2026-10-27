# 파이보에서 easydetect 모델 쓰기 — 연동 지시서

파이보(라즈베리파이 4B) 쪽 코드(openpibo, 파이브레인)에서 사물 인식을 easydetect
ed 모델로 바꾸는 작업 지시서다.

## 목표

- **yolo11s는 지운다.** 코드(YoloOnnx와 그 호출)와 `yolo11s.onnx` 파일 모두.
  되돌릴 곳으로 남기지 않는다. 사물 인식은 ed 하나로 간다.
- 모델은 **`.onnx` 파일 하나**다. 두 종류를 같은 방법으로 읽는다.
  1. **기본 COCO 모델**: COCO 80 클래스, Hugging Face `leeyunjai/themaker`의
     `.model/object/ed-s-320.onnx`. 지금 yolo11s.onnx가 있던 자리다.
  2. **lab에서 학습한 모델**: easydetect lab의 *다운로드 → Pibo로 보내기*로 받은
     `pibo-job<번호>.onnx`.
- 설정으로 어느 파일을 쓸지 고른다. 비어 있으면 기본 COCO 모델이다.
- 더 좋은 COCO 모델이 학습되면 같은 이름의 파일만 바뀐다. 코드는 그대로다.

## 1. 모델 파일

- ONNX opset 17, 입력 이름은 `images`, 입력 크기는 파일마다 고정이다.
- **크기와 모델 종류는 파일마다 다르다.** 파이보용 기본은 ed-s 320이지만, GPU가
  있는 PC에서 ed-m·ed-l 640 등으로 학습한 lab 모델도 같은 형식이다. 입력 크기는
  파일(`session.get_inputs()[0].shape`)에서 읽고, 320이나 특정 모델로 고정하거나
  막지 않는다. 느린 모델을 고르는 건 사용자의 선택이다.
- 파일 안(ONNX metadata)에 들어 있는 것. onnxruntime으로 이렇게 읽는다:
  `session.get_modelmeta().custom_metadata_map`

  | 키 | 값 | 없으면 |
  | --- | --- | --- |
  | `easydetect.names` | 클래스 이름 JSON. 예: `{"0": "person", "1": "컵"}`. 한글일 수 있다 | 이름 대신 번호를 쓴다 |
  | `easydetect.conf` | 이 모델의 확신 기준. 문자열 숫자, 예: `"0.45"`. lab이 검증 사진에서 F1이 가장 높은 값을 넣는다 | 0.5를 쓴다 |

- 기본 COCO 모델 파일은 PC에서 `easydetect/tools/pibo_coco.py`로 만든다.

## 2. 필요한 것

- Python 3.6 이상, numpy, opencv(cv2), **onnxruntime 1.13 이상** (opset 17).
- torch, easydetect, Ultralytics는 필요 없다.
- 64비트 OS(`uname -m`이 `aarch64`)여야 pip로 onnxruntime이 깔린다.

## 3. 구현: openpibo 안에 ed 읽기 클래스를 둔다

YoloOnnx가 있던 자리에 ed를 읽는 클래스를 하나 두고, YoloOnnx는 지운다.
이름은 예를 들어 `EdOnnx`로 한다. 입출력 사양은 4절이다.

지금 사물 인식이 YoloOnnx를 부르던 모양 그대로 부를 수 있게 한다:

- `EdOnnx(model_path, threads=4, conf=None, iou=0.7)`
- `det(img, conf=..., iou=...)` → `(boxes (N,4) x1 y1 x2 y2 픽셀, scores (N,), classes (N,) int)`. 점수 높은 순이다.
- `det.name_of(cls_id)` → 파일 안의 이름

conf와 iou는 클래스 쪽에서 정한다.
- **conf:** 생성할 때 주지 않으면 파일 안의 `easydetect.conf`를 쓴다(없으면 0.5).
  호출이 넘기는 값보다 이 값을 우선한다. openpibo가 늘 넘기는 0.5 때문에 lab이
  모델마다 고른 기준이 무시되지 않게 하기 위해서다.
- **iou: 0.7.** ed는 중복 상자를 거의 내지 않는다. 겹침 제거는 같은 물체가
  두 번 잡힌 드문 경우만 지우면 된다. 0.4로 하면 말 위의 사람처럼 **실제로 겹친
  서로 다른 물체를 지운다.**

**기준 구현:** easydetect-lab 저장소의 `pibo/detect.py`가 같은 계산을 한다.
같은 파일을 넣고 같은 사진이면 openpibo 구현과 상자·점수가 같아야 한다. 확인
목록에서 이걸 비교한다. 단 openpibo는 이 파일을 import하지 않는다. 내용만 참고한다.

## 4. 입출력 사양

`pibo/detect.py`의 `preprocess`·`postprocess`와 같다. 아래와 다르게 하면 점수가 틀어진다.

- **입력:** `(1, 3, S, S)` float32. S는 모델 입력 크기다.
  - BGR을 RGB로 바꾼다.
  - **비율 유지 없이 그냥 S×S로 resize한다. letterbox(회색 여백) 하지 않는다.**
    YOLO와 다른 점이다.
  - 255로 나눈 뒤 NCHW 순서로 둔다.
- **출력:** 두 개다. 순서에 기대지 말고 마지막 차원으로 구분한다.
  - boxes `(1, Q, 4)`: cx, cy, w, h. 원래 사진에 대한 0~1 비율이다.
  - scores `(1, Q, C)`: 클래스별 점수. 0~1 범위가 아니면 아직 로짓이니 sigmoid를
    한 번 적용한다. 이미 확률이면 다시 적용하지 않는다.
- **후처리:**
  - 질의마다 최고 클래스와 점수를 고른다.
  - `score >= conf`인 것만 남겨 점수순으로 정렬한다.
  - 상자는 `x1 = (cx - w/2) * 원래폭`처럼 계산해 사진 안으로 자른다.
  - 겹침 제거: 앞에서부터 클래스 상관없이 IoU > 0.7인 뒤쪽 상자를 지운다. NMS는 따로 없다.

## 5. openpibo 쪽 할 일

- [ ] YoloOnnx와 yolo11s.onnx를 지우고, 사물 인식이 `EdOnnx`를 쓰게 한다.
  openpibo의 실제 호출 방식(`object_detector`, conf/iou 인자 등)을 먼저 확인하고
  그 모양을 지킨다.
- [ ] 설정으로 모델 파일 경로를 고른다. 비어 있으면 `ed-s-320.onnx`를 쓴다.
- [ ] 파일이 없거나 onnxruntime이 1.13보다 낮거나 로드에 실패하면, 무엇이 문제인지
  알려 주는 오류를 낸다. 다른 모델로 몰래 바꾸지 않는다.
- [ ] 화면에 그릴 때(`detect_object_vis` 등) 한글 이름이 깨지지 않게 PIL로 그린다.
  OpenCV `putText`는 한글을 못 그린다.
- [ ] 모델을 바꿀 때 이전 onnxruntime 세션을 놓아 메모리가 쌓이지 않게 한다.
  파이보 램은 2GB다.

## 6. 성능 참고

| 모델 (320 입력) | COCO mAP50-95 | 파이보 4B 한 장 (ONNX Runtime 1.20, 4스레드, 모델 실행만) |
| --- | --- | --- |
| (참고) 지금까지 쓰던 yolo11s.onnx | 0.365 | 300 ms |
| **ed-s-320 (기본)** | **0.381** | 407 ms |
| ed-s-320-fast (2층, 쿼리 50) | 0.366 | 351 ms |
| lab 수업용 모델 (ed-n 320, 쿼리 100) | 데이터마다 다름 | 약 160 ms |

ed-s-320은 yolo11s보다 정확하고 1.36배 느리다. 정확도를 택했다.

**측정 요청:** 파이보에서 `EdOnnx`로 카메라 프레임을 돌려 안정된 뒤의 ms를
적어서 easydetect 쪽에 알려 준다. `ROADMAP.md` 목표 1의 판정에 쓴다.
`pibo/detect.py`로 재도 된다: `python3 detect.py 0 --model <파일> --every 10`.

## 7. 확인 목록 (완료 기준)

- [ ] yolo11s 코드와 파일이 없다. 저장소와 이미지에서 `yolo`를 검색해 남은 게 없다.
- [ ] 설정 없이 `ed-s-320.onnx`로 `detect_object(img)`가 사람·컵 등을
  이름·점수·상자로 돌려준다.
- [ ] lab에서 받은 `pibo-job<번호>.onnx`(한글 클래스 포함)를 고르면 그 모델로
  인식하고, 이름이 한글로 나오고 화면에도 한글로 그려진다.
- [ ] 입력 크기가 다른 파일(예: 640으로 학습한 lab 모델)도 코드 수정 없이 읽힌다.
- [ ] lab 모델의 확신 기준(`easydetect.conf`)이 쓰인다. openpibo가 0.5를 넘겨도 마찬가지다.
- [ ] 같은 사진에서 openpibo의 `EdOnnx`와 `pibo/detect.py`가 같은 상자·점수를
  낸다. 점수는 소수 둘째 자리까지, 상자는 1픽셀 안이어야 한다.
- [ ] 없는 파일을 고르면 이유가 담긴 오류가 난다.
- [ ] 모델을 열 번 바꿔도 메모리가 계속 늘지 않는다.
- [ ] 6절의 파이보 측정값을 보고한다.

## 하지 말 것

- ed 모델에 letterbox 전처리, sigmoid 두 번 적용, NMS iou 0.4를 하지 않는다.
- 모델 형식(파일 안의 키)을 파이보 쪽에서 바꾸지 않는다. 바꿀 게 있으면
  easydetect에 알린다.
- yolo11s로 되돌아가는 길을 남기지 않는다.
