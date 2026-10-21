// easydetect lab — 한국어 / English.
//
// Korean is the source text; English comes from EN below. Text in the page
// (markup, or HTML that script writes) is swapped whole: a text node or a
// title/placeholder whose trimmed text is a key. Strings the script builds
// with values in them go through T`…${x}…`, whose holes are {0}, {1}, … in
// the key. Progress lines the server writes in Korean fall back to FRAGMENTS.
// Anything without an entry stays Korean rather than breaking.
(() => {
  const EN = {
    // header, steps, panels
    "데이터셋": "Datasets", "학습": "Train", "추론": "Predict", "모델": "Models",
    "한국어 / English": "한국어 / English",
    "PyTorch가 이 GPU를 못 써요. 학습이 CPU로 돌아요. CUDA 빌드 torch를 깔아 주세요 (RTX 50 계열은 cu128 이상).":
      "PyTorch can't use this GPU, so training runs on the CPU. Install a CUDA build of torch (cu128 or newer for the RTX 50 series).",
    "데이터 올리기": "Add data", "상자 그리기": "Draw boxes", "모델 쓰기": "Use the model",
    "쉬는 중": "Idle", "CPU 전용": "CPU only", "#{0} 실행 중": "#{0} running", "대기 {0}": "{0} queued",
    "사용률": "Load", "메모리": "Memory",

    // datasets
    "사진·영상·zip을 올리고, 상자를 그린 뒤 학습에 써요.": "Add photos, video or a zip, draw boxes, then train on it.",
    "이미지": "Images", "영상": "Video", "폴더": "Folder",
    "이미지를 끌어다 놓거나 눌러요": "Drop images here, or click", "여러 장 한 번에": "As many as you like",
    "영상을 끌어다 놓거나 눌러요": "Drop a video here, or click", "일정 간격으로 프레임을 뽑아요": "Frames are taken at an interval",
    "프레임 간격": "Every n frames", "최대 장수": "Max frames",
    "zip을 끌어다 놓거나 눌러요": "Drop a zip here, or click",
    "Roboflow 등 YOLO 형식, 또는 images/ + labels/": "YOLO format as Roboflow exports it, or images/ + labels/",
    "이 서버의 폴더 경로": "Folder path on this server", "클래스": "Classes",
    "쉼표로 나눠요. 비우면 data.yaml을 읽거나 알아서 정해요.": "Comma-separated. Leave empty to read data.yaml or work them out.",
    "이름": "Name", "자동": "Automatic", "넣을 곳": "Add to", "새 데이터셋": "New dataset",
    "데이터 추가": "Add data", "내 데이터셋": "My datasets",
    "카드 메뉴에서 '병합 선택'으로 두 개 이상 골라요": "Tick 'Select to merge' in two or more card menus",
    "선택 병합": "Merge selected", "선택 병합 ({0})": "Merge selected ({0})",
    "{0}개 파일 · {1} 외": "{0} files · {1} and more",
    "이미지 {0}": "{0} images", "라벨 {0}/{1}": "labelled {0}/{1}", "라벨 {0}": "{0} labelled",
    "{0}장 추가": "{0} added", "클래스 {0}": "classes {0}",
    "더 보기": "More", "통계 보기": "Statistics", "전체 자동 라벨": "Auto-label all",
    "zip으로 내보내기": "Export as zip", "복제": "Duplicate", "병합 선택": "Select to merge", "삭제": "Delete",
    "라벨": "Labelled", "이 데이터로 학습": "Train on this",
    "상자를 그려야 학습할 수 있어요": "Draw some boxes first",
    "아직 데이터셋이 없어요.": "No datasets yet.", "위에서 사진이나 zip을 올려 봐요.": "Add photos or a zip above to start.",
    "상자 합계": "Boxes in total",
    "라벨 없음 {0}장": "{0} unlabelled", "빈 라벨(배경) {0}장": "{0} empty (background)",
    "아주 작은 상자 {0}개": "{0} tiny boxes", "클래스 범위 밖 {0}개": "{0} boxes with an unknown class",
    "병합된 데이터셋 이름": "Name for the merged dataset",
    "{0}장을 합쳤어요 · 클래스 {1}": "Merged {0} images · classes {1}",
    "데이터셋 #{0}과 그 작업 기록을 지울까요?": "Delete dataset #{0} and its jobs?",
    "이 데이터셋으로 학습한 {0}개도 목록에서 사라져요: {1}. ": "The {0} run(s) trained on it leave the list too: {1}. ",
    "가중치 파일은 runs 폴더에 남아요.": "Their weight files stay in the runs folder.",
    "서버에 닿지 않아요. lab이 켜져 있는지 확인해 주세요": "Cannot reach the server. Is the lab still running?",
    "Pibo용 zip을 받았어요": "Downloaded the Pibo zip",
    "저장하지 못했어요: {0}": "Could not save: {0}",
    "파일이 너무 커요: {0} MB (최대 {1} MB)": "Too large: {0} MB (at most {1} MB)",
    "이 사진의 상자를 저장하지 못했어요 ({0}).\n저장하지 않고 넘어갈까요?": "The boxes on this picture could not be saved ({0}).\nMove on without saving them?",
    "기다리는 중이에요. #{0} 다음에 앞선 {1}개가 끝나면 시작해요": "Waiting: starts after #{0} and the {1} queued before it",
    "기다리는 중이에요. 앞선 {0}개가 끝나면 시작해요": "Waiting: starts after the {0} queued before it",
    "이미지를 골라 주세요": "Pick some images", "영상을 골라 주세요": "Pick a video", "zip을 골라 주세요": "Pick a zip",
    "폴더 경로를 적어 주세요": "Enter a folder path", "올리는 중…": "Uploading…", "복제했어요": "Duplicated",

    // training
    "COCO 가중치에서 시작해 내 데이터에 맞춰요. 끝나면 OpenVINO IR과 ONNX도 만들어 둬요.":
      "Starts from the COCO weights and fits them to your data. An OpenVINO IR and an ONNX are exported at the end.",
    "상자를 그린 데이터셋이 아직 없어요.": "No dataset has boxes yet.", "데이터셋 추가하기": "Add a dataset",
    "목표": "Goal", "빠르게": "Fast", "균형": "Balanced", "정확하게": "Accurate", "수업용 (CPU)": "Classroom (CPU)",
    "노트북 CPU로 사진 300장을 15분 안에 배우도록 맞췄어요. 300장이면 검증용 20%를 빼고 240장 × 35에폭 = 8,400장을 봐요. 4코어 CPU에서 초당 11~15장이라 9~13분, 에폭마다 하는 검증까지 11~15분이에요. 사진이 적으면 에폭을 늘려요(100장이면 100에폭): 적은 사진은 더 여러 번 봐야 잘 배우고, 걸리는 시간은 비슷해요. 25에폭 동안 나아지지 않으면 일찍 끝나요. 더 느린 노트북이면 에폭을 줄이세요.":
      "Set to learn 300 pictures in under 15 minutes on a laptop CPU: 20% are kept for validation, so 240 × 35 epochs = 8,400 pictures. A 4-core CPU does 11-15 a second, so 9-13 minutes, 11-15 with each epoch's validation. Fewer pictures get more epochs (100 epochs for 100 pictures): a small set needs more passes to learn, and the time stays about the same. It stops early after 25 epochs without improvement. On a slower laptop, lower the epochs.",
    "쿼리 수": "Queries", "디노이징 쿼리": "Denoising queries", "자동 (300)": "auto (300)", "자동 (100)": "auto (100)",
    "줄이면 한 스텝이 가벼워져요. 100 · 30이면 CPU에서 약 2배 빨라요. 한 사진에 물체가 몇 개뿐일 때 알맞고, 모델은 이 쿼리 수로 돌아요(내보낸 파일도).":
      "Fewer make each step lighter: 100 · 30 trains about 2× faster on a CPU. Suits pictures with a handful of objects; the model then runs with this many queries (the exported files too).",
    "쿼리 {0} · 디노이징 {1}": "queries {0} · denoising {1}",
    "Pibo로 보내기": "Send to Pibo",
    "model.onnx · labels.txt · detect.py — numpy, OpenCV, onnxruntime만으로 돌아요": "model.onnx · labels.txt · detect.py — runs with numpy, OpenCV and onnxruntime alone",
    "Pibo용 zip을 만드는 중이에요. 몇 초 걸려요": "Making the Pibo zip; a few seconds",
    "시작 모델": "Start from", "n 가장 가벼움 · s 기본 · m · l · x 가장 정확": "n lightest · s default · m · l · x most accurate",
    "비워 두면 easydetect가 정해요.": "Left empty, easydetect decides.",
    "겹친 상자 정리 (IoU)": "Overlapping boxes (IoU)", "끄기": "off",
    "물체 윤곽 칠하기 (세그멘테이션)": "Outline objects (segmentation)",
    "상자마다 MobileSAM이 물체 모양을 칠해요. 내 데이터로 학습한 모델도 마스크 라벨 없이 돼요. 처음 한 번 44MB를 받고, CPU에서는 사진마다 0.2초쯤 더 걸려요.":
      "MobileSAM fills in each box's object. Works for models trained on your own boxes too, no mask labels. Downloads 44 MB once; on a CPU each picture takes about 0.2 s more.",
    "클래스 편집": "Edit class", "이름 바꾸기": "Rename", "로 합치기": "Merge into the one picked",
    "클래스 삭제": "Delete class", "새 이름": "New name",
    "'{0}' 클래스를 지울까요? 상자 {1}개가 함께 지워지고, 뒤 클래스 번호가 하나씩 당겨져요.":
      "Delete the class '{0}'?\nIts {1} boxes go with it, and the classes after it move up one number.",
    "'{0}' 상자 {1}개를 '{2}'(으)로 옮기고 '{3}' 클래스를 지울까요?":
      "Move the {1} '{0}' boxes to '{2}' and delete the class '{3}'?",
    "되돌릴 수 없어요. 필요하면 먼저 데이터셋을 복제해 두세요.":
      "This cannot be undone. Duplicate the dataset first if you may want it back.",
    "이만큼 넘게 겹친 상자는 점수 높은 것 하나만 남겨요. 1이면 꺼요.":
      "Boxes overlapping more than this keep only the highest score. 1 turns it off.",
    "같은 물체의 조각 상자 합치기": "Merge pieces of one object",
    "가려진 물체가 전체와 조각으로 여러 번 잡힐 때, 같은 종류 상자 안에 80% 넘게 들어간 상자를 하나로 정리해요. 그런 장면에서만 켜요: 겹쳐 있는 진짜 물체(군중, 책장)도 지워서 평균으로는 손해예요.":
      "When a half hidden object comes back as the whole and its pieces, a box with over 80% of it inside another of its class is merged into one. Turn it on for such scenes only: it also removes real objects that overlap (crowds, shelves), a loss on average.",
    "학습률": "Learning rate", "자동: 배치 {0} → {1}": "auto: batch {0} → {1}",
    "백본 학습률 비율": "Backbone learning-rate share",
    "학습률에 곱해요. 자동은 모델 크기별 (n·s 0.5, m 0.1, l 0.05, x 0.01)":
      "Multiplies the learning rate. Auto goes by model size (n·s 0.5, m 0.1, l 0.05, x 0.01)",
    "워밍업 에폭": "Warmup epochs", "혼합 정밀도 (GPU만, 빨라요)": "Mixed precision (GPU only, faster)",
    "배치 {0} 기준 자동": "auto for batch {0}",
    "에폭": "Epochs", "배치": "Batch", "고급 설정": "Advanced", "입력 크기": "Input size", "검증 비율": "Validation share",
    "백본 고정": "Freeze backbone", "예 (빨라요)": "Yes (faster)", "아니오": "No", "장치": "Device", "학습 시작": "Start training",
    "{0}에폭": "{0} epochs", "배치 {0}": "batch {0}",
    "안 나아지면 멈추기": "Stop when it stops improving", "에폭 동안 mAP가 그대로면 멈춰요": "epochs with no better mAP, then stop",
    "가장 좋았던 가중치가 남아요. 0이면 끝까지 해요.": "The best weights are kept. 0 runs every epoch.",
    "{0}에폭 정체 시 멈춤": "stop after {0} flat epochs",
    "직접 바꾼 칸 {0}개는 목표를 바꿔도 그대로예요.": "The {0} field(s) you changed stay as you set them.",
    "목표값으로 되돌리기": "Use the goal's values",
    "예 — 빠르고 적은 데이터에 알맞아요": "Yes — faster, suits a small dataset",
    "아니오 — 백본까지 내 데이터에 맞춰요": "No — the backbone learns your data too", "증강": "augment",
    "확대·축소·자르기 증강": "Zoom, crop and colour augmentation",
    "크기와 위치가 다른 사진을 만들어 배워요. 에폭이 많을 때 도움이 되고, 짧은 학습에서는 느려져요.":
      "Trains on pictures at other sizes and positions. Helps longer runs; slows short ones.",
    "데이터셋을 골라 주세요": "Pick a dataset",
    "이 데이터셋은 #{0}이(가) 벌써 대기 중이에요. 하나 더 넣을까요?": "#{0} is already queued for this dataset.\nQueue another?",
    "이 데이터셋은 #{0}이(가) 벌써 학습 중이에요. 하나 더 넣을까요?": "#{0} is already training on this dataset.\nQueue another?",
    "#{0}을(를) 대기열에 넣었어요": "#{0} queued",

    // predict
    "모델을 데이터셋·폴더·영상·웹캠에 돌려 봐요.": "Run a model over a dataset, a folder, a video or a webcam.",
    "확신 정도 (conf)": "Confidence (conf)", "낮추면 더 많이 찾고, 높이면 확실한 것만 남아요.": "Lower finds more; higher keeps only the sure ones.",
    "대상": "Source", "웹캠": "Webcam", "/data/새-촬영분": "/data/new-shots",
    "상자를 그린 mp4로 돌려받아요": "You get an mp4 back with the boxes drawn",
    "카메라 시작": "Start camera", "정지": "Stop", "추론 실행": "Run",
    "데이터셋이 없어요": "There is no dataset", "보내는 중…": "Sending…", "#{0}을(를) 시작했어요": "#{0} started",
    "카메라는 https에서 열려요": "The camera opens over https",
    "처음 뜨는 인증서 경고에서 '고급 → 계속'을 눌러요": "On the certificate warning, choose 'Advanced → Continue' once",
    "카메라는 https나 localhost에서만 열려요. 서버를 --https-port 와 함께 켜 주세요.":
      "Browsers only open a camera on https or localhost. Start the server with --https-port.",
    "카메라가 안 열려요: {0}": "The camera didn't open: {0}", "찾은 물체가 없어요": "Nothing found",
    "내 학습 결과": "My runs", "등록된 모델": "Registered models", "COCO 사전학습 · 80 클래스": "COCO pretrained · 80 classes",

    // models
    "학습 결과에 이름을 붙여 두면 자동 라벨과 추론에서 골라 써요.": "Name a run to pick it for auto-labelling and prediction.",
    "외부 모델 올리기": "Add a model from elsewhere", "IR은 두 파일을 함께 골라요": "Pick both files of an IR",
    "이름 (비우면 파일 이름)": "Name (default: the file name)", "올리기": "Upload",
    "코드에서 쓰기": "Use it in code", "어느 PC든": "On any PC,",
    "하나면 돼요. 추론에는 PyTorch가 필요 없어요. 학습한 모델의 경로가 채워진 코드는 작업을 열면 나와요.":
      "is all it takes; prediction needs no PyTorch. Open a job for code with your model's path filled in.",
    "복사": "Copy", "복사했어요": "Copied", "코드": "Code",
    "등록한 모델이 아직 없어요.": "No models registered yet.", "작업을 열고 \"모델로 등록\"을 눌러요.": "Open a job and press \"Register model\".",
    "이 모델을 목록에서 뺄까요? 파일은 그대로 남아요.": "Remove this model from the list? Its files stay.",
    "파일을 골라 주세요": "Pick a file", "올렸어요": "Uploaded", "모델 이름": "Model name",

    // the job panel
    "시작하기": "Getting started", "닫기": "Close",
    "네 단계면 내 물건을 찾는 모델이 생겨요. 작업을 누르면 여기서 결과를 봐요.":
      "Four steps to a model that finds your objects. Click a job to see it here.",
    "사진·영상·Roboflow zip": "Photos, video, a Roboflow zip", "자동 라벨로 초안을 받아요": "Auto-label drafts them for you",
    "COCO에서 시작해 몇 분": "Minutes, starting from COCO", "추론 · 배포": "Predict · deploy", "웹캠·영상 · OpenVINO IR": "Webcam, video · OpenVINO IR",
    "중지": "Stop", "이어서 학습": "Train more", "평가": "Evaluate", "모델로 등록": "Register model", "다운로드": "Download",
    "로그": "Log", "결과 이미지 zip": "Result images (zip)", "결과 영상": "Result video",
    "이 작업과 결과 파일을 지워요": "Deletes this job and its files",
    "학습 곡선": "Training curve", "에폭별 기록": "Per epoch", "에폭별 기록 ({0})": "Per epoch ({0})", "초": "s",
    "이 모델로 시험해 보기": "Try this model", "이미지 한 장": "One image", "끌어다 놓거나 눌러요": "Drop it here, or click",
    "평가 결과": "Evaluation", "학습 정보": "Run details", "명령줄": "Command line", "다른 PC": "Another PC",
    "저장소": "Repository", "저장소 안 폴더": "Folder in the repository",
    "best.pt도 넣기 — 받은 사람이 이어서 학습할 수 있어요 (용량이 늘어요)": "Include best.pt — so others can keep training it (larger)",
    "업로드용 zip 받기": "Download the upload zip", "README 보기": "Show README", "업로드 명령 보기": "Show upload commands",
    "작업": "Jobs", "전체": "All", "자동 라벨": "Auto-label", "{0}개": "{0}",
    "학습·자동 라벨·평가·추론이": "Training, auto-labelling, evaluation", "여기 차례로 쌓여요.": "and predictions line up here.",
    "이 종류의 작업은 아직 없어요.": "No jobs of this kind yet.",
    "대기": "Queued", "실행 중": "Running", "내보내는 중": "Exporting", "완료": "Done", "실패": "Failed", "중지됨": "Stopped",
    "{0} 걸림": "took {0}", "곧 시작해요": "Starting soon", "기다리는 중이에요. #{0}이(가) 끝나면 시작해요": "Waiting for #{0} to finish",
    "몇 에폭 더 학습할까요?": "How many more epochs?",
    "\"{0}\"(으)로 등록했어요. 모델 탭과 추론에서 골라요": "Registered as \"{0}\". Pick it under Models and Predict.",
    "#{0} 작업과 결과 파일(가중치, IR, 로그)을 지울까요?": "Delete job #{0} and its files (weights, IR, log)?",
    "이어서 학습·평가 {0}개도 같이 지워져요.": "Its {0} continuations and evaluations go with it.",
    "시험할 이미지를 골라 주세요": "Pick an image to try", "추론 중…": "Predicting…",
    // run details
    "시작 가중치": "Started from", "학습 데이터": "Training data", "검증 데이터": "Validation data",
    "학습하는 파라미터": "Trained parameters", "옵티마이저": "Optimizer", "스케줄": "Schedule", "증강": "Augmentation",
    "조기 종료": "Early stopping", "최고 에폭": "Best epoch", "마지막 loss": "Last loss", "버전": "Versions",
    "COCO 사전학습 가중치": "COCO pretrained weights", "ImageNet 백본만 (COCO를 못 받음)": "ImageNet backbone only (COCO unavailable)",
    "처음부터 (무작위)": "From scratch (random)", "직접 준 네트워크": "A network you passed in", "이전 학습 · {0}": "Earlier run · {0}",
    "{0}장 · 상자 {1}": "{0} images · {1} boxes", "배경 {0}": "{0} background", "파라미터 {0}": "{0} parameters",
    "{0} 고정": "{0} frozen", "백본 {0}": "backbone {0}", "일찍 멈춤": "stopped early", "에폭당 {0}": "{0} per epoch",
    "{0}에폭부터": "from epoch {0}", "모델 구조": "Architecture", "이어서 학습한 곳": "Resumed", "학습 상자": "Train boxes", "검증 상자": "Val boxes",
    // results
    "상자": "Boxes", "상자 {0}": "{0} boxes", "지금까지 그린 결과 {0}장": "{0} results drawn so far",
    "처음 {0}장만 보여요. 전부는 다운로드 → 결과 이미지 zip": "Showing the first {0}. All of them: Download → Result images (zip)",
    "초록 상자가 정답, 색 상자가 예측이에요. 빨간 숫자는 놓친 이미지예요.": "Green boxes are the truth, coloured ones the prediction. Red counts mark missed images.",
    "정답 {0}": "truth {0}", "찾음 {0}": "found {0}",
    "{0}시간 {1}분": "{0} h {1} min", "{0}분 {1}초": "{0} min {1} s", "{0}초": "{0} s",
    // code comments
    "상자를 그린 이미지": "the picture with its boxes", "영상 · 웹캠(0) · RTSP: 한 프레임씩 받기": "video · webcam (0) · RTSP: one frame at a time",
    "이미지 · 폴더 · 영상 — 결과는 runs/detect/predict/": "image · folder · video — results in runs/detect/predict/",
    "웹캠 창으로 (q 또는 Esc로 종료)": "in a webcam window (q or Esc quits)",
    "다운로드 → OpenVINO IR 로 zip을 받아 풀어요 ({0} · .bin · labels.txt)": "Download → OpenVINO IR, and unzip it ({0} · .bin · labels.txt)",
    "그 PC에는 추론용만 설치해요 — PyTorch 없이 돼요": "on that PC install the inference package only — no PyTorch needed",
    "라즈베리파이처럼 Intel이 아닌 PC는 다운로드 → ONNX zip ({0} · labels.txt)": "on a non-Intel PC such as a Raspberry Pi: Download → ONNX zip ({0} · labels.txt)",
    "ONNX zip이면": "with the ONNX zip",
    "파이썬에서": "in Python", "Intel NPU면 \"NPU\"": "\"NPU\" on an Intel NPU",
    "\"업로드용 zip 받기\" → 풀면 {0}/ 폴더 (모델 · labels.txt · README.md)": "\"Download the upload zip\" → unzips to {0}/ (model · labels.txt · README.md)",
    "한 번만: 쓰기 권한 토큰으로 로그인": "once: log in with a token that can write",
    "폴더째 올리기 — 저장소의 {0}/ 로 들어가요": "upload the folder — it lands in {0}/ in the repo",
    "받는 쪽 (README에도 같은 코드가 들어 있어요)": "on the receiving end (the README has this code too)",
    "처음 한 번 가중치를 받아요 (COCO 80 클래스)": "downloads the weights once (COCO, 80 classes)",
    "학습한 모델은 그 경로를 그대로 넣어요": "a model you trained: pass its path",

    // label page
    "상자 그리기 — easydetect": "Draw boxes — easydetect", "{0} — 상자 그리기": "{0} — draw boxes",
    "데이터 목록으로": "Back to the datasets", "클래스 추가": "Add a class", "추가": "Add", "폴더별로 보기": "Show one folder",
    "상자 없는 것만 보기": "Only images without boxes", "이전 (←)": "Previous (←)", "이전": "Previous",
    "다음 (→)": "Next (→)", "다음": "Next", "모델이 상자 초안을 그려요": "A model drafts the boxes",
    "이전 이미지의 상자를 그대로 가져와요 (C)": "Copy the previous image's boxes (C)", "이전 상자 복사": "Copy previous boxes",
    "되돌리기": "Undo", "모두 지우기": "Clear all", "이 이미지와 상자를 데이터셋에서 지워요": "Removes this image and its boxes from the dataset",
    "이미지 삭제": "Delete image",
    "끌어서 그려요 · 상자 안을 끌면 옮겨요 · 모서리를 끌면 크기가 바뀌어요 ·": "Drag to draw · drag inside a box to move it · drag a corner to resize ·",
    "클래스 ·": "class ·", "지우기 ·": "delete ·", "되돌리기 ·": "undo ·", "넘기기 ·": "previous / next ·",
    "이전 상자 복사 · 휠 확대 ·": "copy previous boxes · wheel zooms ·", "+끌기로 화면 이동 · 저장은 저절로 돼요": "+drag pans · saving is automatic",
    "모든 폴더 ({0})": "All folders ({0})", "(최상위)": "(top level)", "상자 {0}개": "{0} boxes",
    "빈 라벨 (배경 이미지)": "Empty label (background image)", "아직 상자가 없어요": "No boxes yet",
    "다 그린 것 {0}/{1}": "done {0}/{1}", "복사할 이전 상자가 없어요": "No previous boxes to copy",
    "자동 라벨 중…": "Auto-labelling…", "자동 라벨이 안 됐어요: {0}": "Auto-label failed: {0}",
    "{0} 을(를) 데이터셋에서 지울까요?": "Remove {0} from the dataset?",
    "라벨 종류": "What to label", "상자": "Boxes", "윤곽": "Outlines", "키포인트": "Keypoints",
    "물체마다 상자 하나": "One box an object", "물체마다 윤곽 (세그멘테이션)": "An outline an object (segmentation)",
    "상자와 그 안의 점들 (포즈)": "A box and points inside it (pose)",
    "물체마다 상자를 그려요.": "Draw a box around each object.",
    "물체마다 윤곽을 그려요. 상자를 그리면 AI(MobileSAM)가 윤곽을 따 주고, 점을 끌어 고쳐요. 학습하면 윤곽(마스크) 모델도 함께 맞춰요.":
      "Outline each object. Draw a box and AI (MobileSAM) traces the outline; drag its points to fix it. Training also fits the outline (mask) model.",
    "상자를 그린 뒤 그 안의 키포인트를 차례로 찍어요. 학습하면 키포인트 모델도 함께 학습해요.":
      "Draw a box, then click its keypoints in order. Training also trains the keypoint model.",
    "편집": "Edit", "점 이름 (한 줄에 하나, 순서대로)": "Point names (one a line, in order)",
    "잇는 선 (한 줄에 둘: 이름 - 이름)": "Lines between them (one a line: name - name)",
    "COCO 사람 17점": "COCO people, 17 points",
    "COCO 사람 17점: 자동 라벨에 COCO 포즈 모델을 바로 써요": "COCO's 17 body points: auto-label can use the COCO pose model straight away",
    "취소": "Cancel", "저장": "Save",
    "왼쪽·오른쪽(left/right, l_/r_, 왼/오른)이 짝지어진 이름은 좌우 뒤집기 학습에서 서로 바뀌어요.":
      "Names that pair up left and right (left/right, l_/r_) swap when training flips a picture.",
    "찍었어요": "Placed", "가려진 점": "Hidden point", "아직 안 찍었어요": "Not placed yet",
    "윤곽 그리기": "Draw outline", "클릭으로 점을 찍어 윤곽을 그려요 (P)": "Click points to draw an outline (P)",
    "AI 윤곽": "AI outline", "고른 상자(없으면 윤곽 없는 상자 모두)의 윤곽을 AI가 따요 (S)":
      "AI traces the selected box's outline, or every box without one (S)",
    "상자를 그리면 바로": "As soon as a box is drawn",
    "윤곽 그리기: 클릭으로 점 · 첫 점이나": "draw outline: click for points · the first point or",
    "로 닫기 ·": "closes ·", "취소 ·": "cancels ·",
    "AI 윤곽 · 점을 끌어 고치기 · 선을 더블클릭하면 점 추가 · 점 우클릭은 지우기":
      "AI outline · drag points to fix · double-click a line to add a point · right-click a point to remove it",
    "상자를 고르고 클릭으로 키포인트를 차례로 찍어요 ·": "Pick a box and click its keypoints in order ·",
    "+클릭은 가려진 점 ·": "+click: a hidden point ·", "다음 점 · 점을 끌어 옮기기 · 점 우클릭: 가려짐 → 지우기 ·":
      "next point · drag a point to move it · right-click a point: hidden → removed ·",
    "선택 해제": "deselect",
    "키포인트를 모두 지우고 상자만 남겨요 (라벨 파일 {0}개).\n되돌릴 수 없어요. 필요하면 먼저 데이터셋을 복제해 두세요.":
      "This removes every keypoint and keeps the boxes ({0} label files).\nIt cannot be undone; duplicate the dataset first if you may want them.",
    "잇는 선을 못 읽었어요: {0}": "Could not read the line: {0}",
    "키포인트 이름을 하나 이상 적어 주세요": "Name at least one keypoint", "같은 이름이 두 번 있어요": "A name appears twice",
    "윤곽 {0}개는 상자로 바뀌고": "{0} outlines become boxes", "키포인트 {0}종은 지워져요": "{0} keypoints are removed",
    "라벨 파일 {0}개를 고쳐 써요: {1}.\n되돌릴 수 없어요. 필요하면 먼저 데이터셋을 복제해 두세요.":
      "{0} label files are rewritten: {1}.\nIt cannot be undone; duplicate the dataset first if you may want them.",
    "윤곽을 딸 상자가 없어요": "No box to outline", "윤곽 따는 중…": "Tracing outlines…",
    "윤곽을 못 땄어요: {0}": "Could not trace outlines: {0}",
    "윤곽 없는 상자 {0}개": "{0} boxes without an outline", "키포인트 없는 상자 {0}개": "{0} boxes without keypoints",
    "키포인트 찍기 (포즈)": "Place keypoints (pose)",
    "윤곽·키포인트로 학습한 모델은 그 학습에서 맞춘 윤곽·키포인트 모델을 써요. 아니면 키포인트는 COCO 사람 17점이에요.":
      "A model trained on outlines or keypoints uses the outline or keypoint model fitted with it; otherwise keypoints are COCO's 17 body points.",
    "상자를 학습한 뒤 윤곽으로 MobileSAM의 마스크 디코더도 맞춰요 (기본 20에폭). 학습 전보다 나빠지지 않아요.":
      "After the boxes, MobileSAM's mask decoder is fitted to the outlines (20 epochs). It never ends worse than it started.",
    "상자를 학습한 뒤 상자 안 키포인트 모델도 같은 에폭만큼 학습해요. 그만큼 시간이 더 걸려요.":
      "After the boxes, a keypoint model trains for as many epochs on what is inside them, which takes longer.",

    // export, and what the server refuses
    "INT8도 만들기 (CPU에서 약 2배 빠름, 정확도 확인 필요)": "Also make INT8 (about 2× faster on a CPU; check its accuracy)",
    "사진 목록이 바뀌었어요. 페이지를 새로 고쳐 주세요": "The picture list has changed. Reload the page",
    "zip 파일이 없어요": "No zip uploaded", "zip 파일이 아니에요": "That file is not a zip",
    "올린 파일이 없어요": "No files uploaded", "이미지 파일이 하나도 없어요": "None of those files are images",
    "올린 영상이 없어요": "No video uploaded", "영상을 읽지 못했어요": "Could not read that video",
    "영상에 프레임이 없어요": "That video had no frames", "이미지를 읽지 못했어요": "Could not read that image",
    "키포인트마다 이름을 적어 주세요 (하나 이상)": "Name every keypoint, and give at least one",
    "같은 이름의 키포인트가 두 개 있어요": "Two keypoints have the same name",
    "합칠 데이터셋을 두 개 이상 골라 주세요": "Pick at least two datasets to merge",
    "클래스 이름을 하나 이상 적어 주세요": "Give at least one class name",
    "지울 클래스 번호를 주세요": "Give the class index to remove", "그런 클래스가 없어요": "No such class",
    "있는 다른 클래스로 합쳐 주세요": "Merge into another existing class",
    "데이터셋에는 클래스가 하나는 있어야 해요": "A dataset keeps at least one class",
    "이 작업에는 가중치가 없어요": "That job has no weights", "가중치 파일이 없어요": "No weights on disk",
    ".pt, .onnx, 또는 IR의 .xml과 .bin을 올려 주세요": "Expected a .pt, a .onnx, or an IR's .xml and .bin",
    "OpenVINO IR은 파일 두 개예요. .xml과 .bin을 함께 골라 주세요": "An OpenVINO IR is two files — pick the .xml and its .bin together",
    "그런 모델이 없어요": "No such model", "데이터셋, 폴더 경로, 영상 중 하나를 주세요": "Give a dataset, a folder path or a video",
    "이 데이터셋에는 아직 라벨이 없어요": "That dataset has no labels yet",
    "patience는 0(끄기) 이상이어야 해요": "patience must be 0 (off) or more",
    "그런 작업이 없어요": "No such job", "그런 학습 작업이 없어요": "No such training job",
    "작업이 아직 돌고 있어요": "That job is still going", "이어서 학습할 last.pt가 없어요": "No last.pt to resume from",
    "아직 평가하지 않았어요": "Not evaluated yet", "추론 결과가 없어요": "No prediction results",
    "그런 이미지가 없어요": "No such image", "아직 받을 파일이 없어요": "Nothing to download yet",
    "가중치가 없어요": "No weights", "내보낸 IR이 없어요": "No exported IR", "내보낸 ONNX가 없어요": "No exported ONNX",
    "아직 결과가 없어요": "No results yet", "결과 영상이 없어요": "No annotated video", "로그가 없어요": "No log",
    "모르는 파일 종류예요": "Unknown file kind", "공유할 모델은 학습 작업에만 있어요": "Only a training run has a model to share",
    "내보낸 IR이 없어요. 내보내기가 실패했거나 아직 학습 중이에요": "No exported IR — the export failed or the run is still going",
    "학습한 모델이 없어요": "No trained model here", "결과 이미지를 만들지 못했어요": "Could not encode the result",
    "올린 이미지가 없어요": "No image uploaded", "그런 데이터셋이 없어요": "No such dataset",
    "이미지를 찾지 못했어요": "No images found",
    "데이터셋마다 키포인트가 달라요. 먼저 같은 키포인트로 맞춰 주세요": "These datasets have different keypoints; give them the same keypoints first",
    "에폭: 정수로 적어 주세요": "Epochs: enter a whole number", "배치: 정수로 적어 주세요": "Batch: enter a whole number",
    "입력 크기: 정수로 적어 주세요": "Input size: enter a whole number",
    "쿼리 수: 정수로 적어 주세요": "Queries: enter a whole number", "디노이징 쿼리: 정수로 적어 주세요": "Denoising queries: enter a whole number", "더 할 에폭: 정수로 적어 주세요": "Extra epochs: enter a whole number",
    "중지했어요": "Stopped",

    // added with the design review: the overview, the panes, the form's checks
    "다섯 단계를 모두 해 봤어요": "You've been through all five steps", "단계 보기": "Show the steps",
    "다음 할 일": "Next", "바로 가기": "Go there", "#{0} 학습 결과": "run #{0}",
    "클래스별 상자": "Boxes per class", "상자 합계 {0}": "{0} boxes in total", "로그 끝부분": "End of the log",
    "놓친 이미지만 보기": "Only missed images", "놓친 이미지가 없어요": "No missed images", "설명": "More",
    "카메라 화면과 찾은 결과는 가운데 칸에 나와요.": "The camera and what it finds show in the middle pane.",
    "비워 두면 easydetect가 정해요. 바꾼 칸은 파란 테두리로 보여요.": "Left empty, easydetect decides. Fields you filled in are outlined in blue.",
    "자동 (1)": "auto (1)", "자동 (0.0001)": "auto (0.0001)", "자동 (0)": "auto (0)",
    "loss (왼쪽)": "loss (left)", "mAP50-95 (오른쪽)": "mAP50-95 (right)",
    "{0}을(를) 숫자로 적어 주세요": "Enter a number for {0}", "{0}은(는) 정수여야 해요": "{0} must be a whole number",
    "{0}은(는) {1}의 배수여야 해요": "{0} must be a multiple of {1}",
    "{0}은(는) {1}에서 {2} 사이여야 해요": "{0} must be between {1} and {2}",
    "{0}은(는) {1} 이상이어야 해요": "{0} must be at least {1}",
    "작업이 멈췄어요. 아래 오류를 확인해 주세요": "The job stopped with an error:",
    "COCO 가중치를 받지 못해 ImageNet 백본에서 시작했어요. 정확도가 많이 낮을 수 있어요. 인터넷 연결을 확인하고 다시 학습해 보세요.":
      "The COCO weights could not be downloaded, so this run started from an ImageNet backbone only. Expect much lower accuracy; check the internet connection and train again.",
    "검증 상자 {0}": "{0} val boxes", "클래스 (낮은 것부터)": "Class (weakest first)",
    "첫 에폭이 끝나면 곡선이 나와요": "The curve appears after the first epoch", "최고 {0}": "best {0}", "에폭 {0}": "Epoch {0}",
    "이 이미지의 물체": "Objects in this image", "축소 (-)": "Zoom out (-)", "확대 (+)": "Zoom in (+)",
    "맞춤": "Fit", "화면에 맞추기 (0)": "Fit to the screen (0)", "단축키 {0}": "Shortcut {0}",
    "상자 없는 다음 이미지 ·": "next image without boxes ·", "확대·축소·맞춤 ·": "zoom in / out / fit ·",
    "이름표 숨기기 ·": "hide name tags ·",
    "아직 없어요. 그림 위를 끌어서 그려요.": "None yet. Drag on the picture to draw one.",
    "상자 {0}개를 찾았어요. 확인하고 고쳐 주세요": "Found {0} boxes. Check and fix them",
    "이름표를 보여요": "Showing name tags", "이름표를 숨겨요 (L)": "Name tags hidden (L)",
    "상자 없는 이미지가 더 없어요": "No more images without boxes",
  };

  // what the server writes while a job runs: "에폭 3/50 · 배치 12/40 · 남은 시간 약 4분"
  const FRAGMENTS = [
    // whole refusals with a value in them, before the word-by-word ones below
    [/^에폭 (\d+)\/(\d+)에서 중지했어요$/, "Stopped at epoch $1/$2"],
    [/^에폭은 1에서 (\d+) 사이여야 해요$/, "Epochs must be between 1 and $1"],
    [/^배치는 1에서 (\d+) 사이여야 해요$/, "Batch must be between 1 and $1"],
    [/^쿼리 수는 (\d+)에서 (\d+) 사이여야 해요$/, "Queries must be between $1 and $2"],
    [/^디노이징 쿼리는 (\d+)에서 (\d+) 사이여야 해요$/, "Denoising queries must be between $1 and $2"],
    [/^Pibo용 zip을 만들지 못했어요: (.+)$/, "Could not make the Pibo zip: $1"],
    [/^더 할 에폭은 1에서 (\d+) 사이여야 해요$/, "Extra epochs must be between 1 and $1"],
    [/^입력 크기는 32의 배수로, (\d+)에서 (\d+) 사이여야 해요$/, "Input size must be a multiple of 32, from $1 to $2"],
    [/^(\S+)은\(는\) ([\d.e+-]+)에서 ([\d.e+-]+) 사이여야 해요$/, "$1 must be between $2 and $3"],
    [/^(\S+)은\(는\) 숫자여야 해요$/, "$1 must be a number"],
    [/^(\S+)은\(는\) 0보다 크고 1 이하여야 해요$/, "$1 must be above 0 and at most 1"],
    [/^(.+)은\(는\) 폴더가 아니에요$/, "$1 is not a folder"],
    [/^(.+)이\(가\) 없어요$/, "$1 does not exist"],
    [/^라벨 종류는 (.+) 중 하나예요$/, "The label kind is one of $1"],
    [/^#(\d+)이\(가\) 이 데이터셋을 쓰고 있어요\. 끝나길 기다리거나 멈춰 주세요$/, "Job #$1 uses this dataset; wait for it or stop it"],
    [/^#(\d+)이\(가\) 아직 이 데이터셋을 쓰고 있어요$/, "Job #$1 is still using it"],
    [/^라벨이 이미 클래스 (\d+)을\(를\) 써요\. 이름을 (\d+)개 이상 두거나 그 상자부터 고쳐 주세요$/,
     "Labels already use class $1; keep at least $2 names or relabel those boxes first"],
    [/^이름이 (.+)인 파일이 너무 많아요$/, "Too many files named $1"],
    [/^zip 안에 위험한 경로가 있어요: (.+)$/, "Unsafe path in the archive: $1"],
    [/(\d+)시간/g, "$1 h"], [/(\d+)분/g, "$1 min"], [/(\d+)초/g, "$1 s"],
    [/남은 시간 계산 중/g, "estimating time left"], [/남은 시간 약 ([^·]+?)\s*$/, "~$1 left"],
    [/키포인트 학습/g, "keypoint training"], [/윤곽\(마스크\) 학습/g, "outline (mask) training"],
    [/준비 · 사진 (\d+)\/(\d+) 읽는 중/g, "preparing · reading picture $1/$2"],
    [/키포인트 AP/g, "keypoint AP"], [/윤곽 mIoU ([\d.]+) \(학습 전 ([\d.]+)\)/g, "outline mIoU $1 (before training $2)"],
    [/에폭 (\d+)\/(\d+)/g, "epoch $1/$2"], [/배치 (\d+)\/(\d+)/g, "batch $1/$2"], [/검증 중/g, "validating"],
    [/프레임 (\d+)\/(\d+)/g, "frame $1/$2"], [/이미지 (\d+)\/(\d+)/g, "image $1/$2"],
    [/상자 (\d+)개/g, "$1 boxes"], [/(\d+)장 미리보기/g, "$1 previews"], [/(\d+)장/g, "$1 images"],
    [/모델 불러오는 중…/g, "loading the model…"], [/#(\d+) 이어서 \+(\d+)에폭/g, "#$1 continued, +$2 epochs"],
    [/이미 학습은 끝났고 내보내는 중입니다/g, "training is over; exporting"], [/^업로드$/, "uploaded"], [/^서버가 다시 켜지면서 끊겼어요$/, "interrupted by a restart"],
    [/^#(\d+)이\(가\) 아직 돌고 있어요\. 먼저 멈춰 주세요$/, "#$1 is still running — stop it first"],
    [/^등록한 모델 '(.+)'이\(가\) 이 결과를 써요\. 모델 탭에서 먼저 빼 주세요$/,
     "registered model '$1' uses this run — remove it under Models first"],
    [/^'(.+)'은\(는\) 랩 폴더 밖이에요\. 서버를 (\w+)=1로 켜면 열 수 있어요$/,
     "'$1' is outside the lab's folders. Start the server with $2=1 to open it"],
    [/^'(.+)'은\(는\) 등록한 모델도 랩 폴더 안의 파일도 아니에요\. 모델 탭에서 올리거나, 서버를 (\w+)=1로 켜 주세요$/,
     "'$1' is neither a registered model nor a file in the lab's folders. Upload it under Models, or start the server with $2=1"],
  ];

  const HANGUL = /[가-힣]/;
  const ATTRS = ["title", "placeholder", "label", "aria-label"];
  const norm = (s) => s.trim().replace(/\s+/g, " ");
  const store = { get: () => { try { return localStorage.getItem("lang"); } catch { return null; } },
                  set: (v) => { try { localStorage.setItem("lang", v); } catch {} } };
  let lang = store.get() || ((navigator.language || "").toLowerCase().startsWith("ko") ? "ko" : "en");

  function english(text) {
    const key = norm(text);
    if (!HANGUL.test(key)) return null;
    let en = EN[key];
    if (en == null) {
      en = key;
      for (const [pattern, to] of FRAGMENTS) en = en.replace(pattern, to);
      if (en === key) return null;
    }
    // keep the spacing around the words: "이미지 " before a <b> stays "Images "
    return text.match(/^\s*/)[0] + en + text.match(/\s*$/)[0];
  }

  // T`이미지 ${n}` or T("쉬는 중")
  function T(strings, ...values) {
    if (typeof strings === "string") strings = [strings];
    const korean = () => strings.reduce((out, s, i) => out + (i ? values[i - 1] : "") + s, "");
    if (lang !== "en") return korean();
    const en = EN[norm(strings.reduce((k, s, i) => k + (i ? `{${i - 1}}` : "") + s, ""))];
    return en == null ? korean() : en.replace(/\{(\d+)\}/g, (_, i) => values[i]);
  }

  const ORIGINAL = new WeakMap();   // text node → its Korean
  function swap(root) {
    // never inside code: an inline <script> the parser has just added is seen
    // here too, and rewriting its source before it runs broke the page in English
    const host = root.nodeType === 1 ? root : root.parentElement;
    if (host?.closest("pre, script, style, [data-no-i18n]")) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => n.nodeType === 1 && (n.closest("pre, script, style, [data-no-i18n]"))
        ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
    for (let node = root.nodeType === 3 ? root : walker.currentNode; node; node = walker.nextNode()) {
      if (node.nodeType === 3) {
        if (!ORIGINAL.has(node)) {
          if (!HANGUL.test(node.nodeValue)) continue;
          ORIGINAL.set(node, node.nodeValue);
        }
        // data-en on the parent settles a word that means two things ("모델": Models / Model)
        const ko = ORIGINAL.get(node), over = node.parentElement?.dataset.en;
        const want = lang === "en" ? over ?? english(ko) ?? ko : ko;
        if (node.nodeValue !== want) node.nodeValue = want;
      } else if (node.nodeType === 1) {
        if (node.closest("pre, script, style, [data-no-i18n]")) continue;
        for (const a of ATTRS) {
          if (!node.hasAttribute(a)) continue;
          const keep = `data-ko-${a}`;
          if (!node.hasAttribute(keep)) {
            if (!HANGUL.test(node.getAttribute(a))) continue;
            node.setAttribute(keep, node.getAttribute(a));
          }
          const ko = node.getAttribute(keep);
          node.setAttribute(a, lang === "en" ? english(ko) ?? ko : ko);
        }
      }
      if (root.nodeType === 3) break;
    }
  }

  function button() {
    const b = document.getElementById("langToggle");
    if (b) b.textContent = lang === "en" ? "한" : "EN";
  }

  function set(next) {
    lang = next; store.set(next);
    document.documentElement.lang = next;
    swap(document.body); button();
    dispatchEvent(new Event("langchange"));   // the page rewrites what its script wrote
  }

  // whatever the page adds later, in English while English is on
  new MutationObserver((records) => {
    if (lang !== "en") return;
    for (const r of records) r.addedNodes.forEach((n) => {
      if (n.nodeType === 3 ? n.parentElement && !n.parentElement.closest("pre, script, style, [data-no-i18n]") : n.nodeType === 1) swap(n);
    });
  }).observe(document.body, { childList: true, subtree: true });

  document.documentElement.lang = lang;
  if (lang === "en") swap(document.body);
  button();
  document.getElementById("langToggle")?.addEventListener("click", () => set(lang === "en" ? "ko" : "en"));

  window.T = T;
  window.i18n = { get lang() { return lang; }, set, english };
})();
