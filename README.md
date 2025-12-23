# seasurface_detector

Консольная утилита на C++17 для детектирования надводных объектов на изображениях/видео, сохранения результатов (bbox + CSV), опционального трекинга (ID) и оценки качества (Precision/Recall) по GT.

Поддерживаются:
- вход: одиночное изображение / директория с изображениями / видео
- выход: изображения/видео с bbox + CSV с детекциями
- инференс: **CPU** и **GPU (CUDA)** через **ONNX Runtime CUDA Execution Provider**

---

## 0) Требования
- Ubuntu 24.04
- CMake ≥ 3.16, компилятор C++17 (g++/clang++)
- OpenCV (dev-пакеты)
- ONNX Runtime (CPU или GPU сборка)
- Для GPU режима: NVIDIA драйвер + CUDA runtime libs + cuDNN 9

---

## 1) Установка зависимостей

### 1.1 Системные пакеты
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake \
  libopencv-dev
```

### 1.2 ONNX Runtime (CPU/GPU)

Проект ожидает ONNX Runtime в:
```
third_party/onnxruntime/
  include/
  lib/
```

Скачайте CPU/GPU(CUDA) сборку ONNX Runtime (tgz), распакуйте в `third_party/onnxruntime`. Версия для GPU уже содержит необходимые пакеты для работы на CPU.

После сборки программы проверьте, что используется нужный ONNX Runtime из third_party:
```bash
ldd build/seasurface_detector | grep onnxruntime
```
Должно указывать на `.../third_party/onnxruntime/lib/libonnxruntime.so...`

После распаковки проверьте зависимости CUDA провайдера (если установлена версия для GPU):
```bash
ldd third_party/onnxruntime/lib/libonnxruntime_providers_cuda.so | grep "not found" || echo "OK: all deps found"
```

Минимальный набор:
- `libcublas.so.12`
- `libcublasLt.so.12`
- `libcurand.s0.10`
- `libcufft.so.11`
- `libcudart.so.12`
- `libcudnn.so.9`

### 1.3 Установка cuDNN 9 (если `libcudnn.so.9 => not found`)
Обычно это делается через репозиторий NVIDIA (через `cuda-keyring`). Важно создать `pin` в `/etc/apt/preferences.d/`, разрешающий скачивание только необходимых пакетов, чтобы пакеты из репозитория NVIDIA не конфликтовали с пакетами из репозитория вашей системы. Необходимо установить пакет `cudnn9-cuda-12`.

### 1.4 NVIDIA драйвер
```bash
nvidia-smi
```
Должна выводиться ваша видеокарта и версия драйвера, без ошибок.

---

## 2) Сборка программы

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
```

Параметр `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` необязателен. Он используется для получения списка комманд компиляции, которые нужны для настройки среды VS Code.

---

## 3) Запуск программы

**ВАЖНО** В этом проекте используется CLI-парсер OpenCV. Аргументы передаются **в формате `--ключ=значение`** (поэтому `bash` не будет разворачивать `~`).

### Примеры

#### A) Одиночное изображение
```bash
./build/seasurface_detector \
  --input=datasets/seadronessee_yolo/val/images/48.jpg \
  --model=models/best.onnx \
  --classes=models/classes.txt \
  --track=false
```

#### B) Директория с изображениями
```bash
./build/seasurface_detector \
  --input=datasets/seadronessee_yolo/val/images \
  --model=models/best.onnx \
  --classes=models/classes.txt \
  --track=false
```

#### C) Видео
```bash
./build/seasurface_detector \
  --input=~/Downloads/ship_video.mov \
  --model=models/best.onnx \
  --classes=models/classes.txt \
  --use_gpu=true
```

---

## 4) Демонстрация работы
![example1](examples/annotated_71.jpg)
![example2](examples/annotated_4867.jpg)

---

## 5) Флаги CLI

```
./build/seasurface_detector
seasurface_detector (C++17/OpenCV) - maritime surface object detection
Usage: seasurface_detector [params] 

        -?, -h, --help, --usage
                print this message
        --class_agnostic (value:true)
                if true, ignore class_id in eval
        --classes
                path to classes.txt (one class name per line)
        --conf (value:0.25)
                confidence threshold
        --eval (value:false)
                compute Precision/Recall (requires --gt_csv)
        --gt_csv
                ground truth csv for evaluation
        -i, --input
                path to image / directory / video (required)
        --imgsz (value:640)
                inference image size (must match ONNX unless dynamic)
        --iou (value:0.5)
                IoU threshold for eval and tracking
        --max_missed (value:30)
                tracker max missed frames
        --model
                path to ONNX model (YOLOv8, requires --classes)
        --nms (value:0.45)
                NMS IoU threshold
        -o, --output (value:out)
                output directory
        --track (value:true)
                enable simple tracking IDs
        --use_gpu (value:false)
                use GPU for DNN inference if available
```

---

## 6) Форматы CSV

### 6.1 CSV детекций (`--csv=...`)
```
source,frame,track_id,class_id,class_name,conf,x,y,w,h
```

### 6.2 GT CSV (`--gt_csv=...`)
Ожидается:
```
source,frame,class_id,x,y,w,h
```

---

## 7) Описание алгоритмов
Раздел описывает алгоритмы, реализованные в проекте.

### 7.1 Карта реализованных алгоритмов и файлов

- **Пайплайн обработки входных данных (изображение/директория/видео) и сохранение результатов** — `src/main.cpp`
- **Детерминированный выбор цвета для bbox по ID/классу и отрисовка bbox + подписи** — `src/main.cpp`
- **Инференс YOLOv8 через ONNX Runtime (CPU/GPU), препроцессинг `letterbox`, декодирование выхода, NMS, обратное отображение координат** — `src/detector.hpp`, `src/detector.cpp`
- **Простой трекинг (присвоение `track_id`) по IoU с механизмом “пропусков” (`max_missed`)** — `src/tracker.hpp`, `src/tracker.cpp`
- **Оценка качества детекций (TP/FP/FN, Precision/Recall) по GT CSV с IoU-сопоставлением** — `src/evaluator.hpp`, `src/evaluator.cpp`
- **Запись результатов детекций в CSV** — `src/csv.hpp`, `src/csv.cpp`
- **Вспомогательные функции (создание директорий, форматирование чисел и т. п.)** — `src/utils.hpp`, `src/utils.cpp`

### 7.2 Пайплайн обработки входных данных и сохранение результатов

**Где реализовано:** `src/main.cpp`

**Назначение:**  
Обработка входного пути (изображение / директория с изображениями / видео), запуск детектора (и опционально трекера), сохранение:
- `dets.csv` с детекциями,
- изображений/видео с нарисованными bbox.

**Входные данные:**  
- `--input`: путь к файлу или директории;
- параметры детектора (`--model`, `--classes`, `--imgsz`, `--conf`, `--nms`, `--use_gpu`);
- параметры трекинга и метрик (`--track`, `--max_missed`, `--iou`, `--eval`, `--gt_csv`, `--class_agnostic`).

**Выходные данные:**  
- CSV: `out/dets.csv`;
- `annotated_*.jpg` / `annotated_*.mp4` в директории `--output`;
- строка со статистикой времени и (при `--eval=true`) метрики Precision/Recall.

**Основные шаги:**
1. Валидация параметров, создание выходной директории.
2. Определение типа входа:
   - директория -> перечисление и сортировка файлов изображений;
   - файл -> попытка трактовать как изображение (по расширению/`imread`), иначе как видео (`VideoCapture`).
3. Для каждого кадра/изображения:
   - инференс детектора `detect(...)`;
   - при включенном трекинге — `tracker.update(...)`;
   - запись каждой детекции в CSV;
   - при включенной оценке — `evaluator.add_frame(...)`;
   - отрисовка bbox и подписи на копии кадра и сохранение.

### 7.3 Препроцессинг `letterbox` (сохранение пропорций + паддинг)

**Где реализовано:** `YoloOnnxDetector::letterbox` — `src/detector.cpp`

**Назначение:**  
Приведение входного изображения к квадрату `imgsz × imgsz` без искажения пропорций: изображение масштабируется с сохранением aspect ratio и размещается по центру на фоне фиксированного цвета.

**Входные данные:**  
- `src`: исходное BGR изображение `H×W`;
- `new_w=new_h=imgsz`.

**Выходные данные:**  
- `out`: изображение `imgsz × imgsz`;
- `scale`: коэффициент масштабирования;
- `pad_w`, `pad_h`: величины отступов слева/сверху.

**Шаги алгоритма:**
1. Вычисление `scale = min(new_w/W, new_h/H)`.
2. Масштабирование до размеров `rw = round(W*scale)`, `rh = round(H*scale)`.
3. Вычисление отступов: `pad_w=(new_w-rw)/2`, `pad_h=(new_h-rh)/2`.
4. Создание холста `imgsz×imgsz` с заливкой `(114,114,114)` и копирование `resized` в центр.

### 7.4 Подготовка входа модели и запуск ONNX Runtime (CPU/GPU)

**Где реализовано:** `YoloOnnxDetector::detect` и инициализация `YoloOnnxDetector` — `src/detector.cpp`

**Назначение:**  
Запуск инференса YOLOv8 ONNX на CPU или GPU (CUDA Execution Provider при наличии), получение результата.

**Основные шаги:**
1. Выполнение `letterbox` и построение входного тензора:
   - `blobFromImage` с нормализацией `1/255`, перестановкой каналов BGR→RGB (`swapRB=true`);
   - итоговая форма: `[1, 3, imgsz, imgsz]`.
2. Создание `Ort::Session`:
   - при `use_gpu=true` выполняется попытка подключения CUDA EP;
   - при ошибке создания с GPU‑настройками выполняется fallback на CPU‑сессию.
3. Запуск `session_->Run(...)` и извлечение тензора результата.
4. Обработка динамической формы результата:
   - если в shape есть одна неопределённая размерность (`<=0`), она восстанавливается по числу элементов тензора.

### 7.5 Декодирование выхода YOLO (bbox + score + class_id)

**Где реализовано:** `YoloOnnxDetector::decode_output` — `src/detector.cpp`

**Назначение:**  
Преобразование тензора выхода модели в набор кандидатов детекций: прямоугольник bbox, confidence score и `class_id`.

**Поддерживаемые раскладки выхода:**  
Алгоритм рассчитан на выход ранга 2 или 3 и пытается привести его к матрице вида `N × attrs`, где `attrs` равно:
- `4 + nc` (без явного objectness), либо
- `5 + nc` (с objectness),
где `nc` — число классов.

**Шаги алгоритма:**
1. Приведение тензора к 2D‑матрице (при необходимости транспонирование) так, чтобы строки соответствовали предсказаниям.
2. Для каждой строки:
   - чтение `(x, y, w, h)`; при эвристическом распознавании нормализованных координат (все ≤ 2) выполняется умножение на `imgsz`;
   - при наличии objectness: `score = obj * max_class_prob`, иначе `score = max_class_prob`;
   - отсев по `conf_thr`;
   - перевод `(x, y, w, h)` из центра в `Rect(left, top, w, h)`.

**Краевые случаи:**
- Некорректная размерность выхода (не 2/3) приводит к исключению.
- Неожиданное число атрибутов приводит к исключению.

### 7.6 Class-wise Non-Maximum Suppression (NMS)

**Где реализовано:** `YoloOnnxDetector::nms_classwise` — `src/detector.cpp`

**Назначение:**  
Удаление дублирующихся bbox на основе IoU с сохранением наиболее уверенных предсказаний. NMS выполняется **отдельно для каждого класса**.

**Шаги алгоритма:**
1. Группировка индексов детекций по `class_id`.
2. Для каждой группы:
   - вызов `cv::dnn::NMSBoxes(...)` с параметрами `conf_thr` и `nms_thr`;
   - перенос индексов “оставленных” bbox в итоговый список.

### 7.7 Обратное отображение координат bbox к исходному изображению

**Где реализовано:** финальная часть `YoloOnnxDetector::detect` — `src/detector.cpp`

**Назначение:**  
Перевод bbox из координат `imgsz × imgsz` (после `letterbox`) обратно в систему координат исходного кадра.

**Шаги алгоритма:**
1. Для каждого bbox после NMS:
   - вычитание отступов: `(x - pad_w, y - pad_h)`;
   - деление на `scale` для возврата к исходному масштабу;
   - округление координат.
2. Ограничение bbox границами кадра (intersection с прямоугольником `0..W, 0..H`).
3. Отсев bbox с нулевой площадью.

### 7.8 Простой трекинг по IoU (назначение `track_id`)

**Где реализовано:** `SimpleTracker::update` — `src/tracker.cpp`

**Назначение:**  
Назначение стабильных идентификаторов объектам между кадрами на основе пересечения bbox (IoU). Используется простой жадный матчинг с учётом “пропусков” кадров.

**Состояние треков:**  
- Для каждого `track_id` хранится bbox и счётчик `missed` (сколько кадров объект не подтверждался).

**Шаги алгоритма:**
1. Увеличение `missed` для всех существующих треков.
2. Для каждой новой детекции:
   - поиск трека с максимальным IoU по всем активным трекам;
   - если `best_iou ≥ iou_threshold`, детекция получает найденный `track_id`, bbox трека обновляется, `missed=0`;
   - иначе создаётся новый трек с новым `track_id`.
3. Удаление треков, у которых `missed > max_missed`.

**Особенности реализации:**  
- Используется жадный поиск “лучшего” трека для каждой детекции без взаимного исключения уже сопоставленных треков, поэтому при близких объектах возможны коллизии назначений.

### 7.9 Оценка качества (TP/FP/FN, Precision/Recall) по GT CSV

**Где реализовано:** `Evaluator` — `src/evaluator.cpp`

**Назначение:**  
Подсчёт качества детекций относительно разметки (GT) в формате CSV по порогу IoU и с опциональным учётом класса.

**Входные данные:**  
- GT CSV: строки `source,frame,class_id,x,y,w,h` (ключ кадра — `source#frame`);
- предсказания `preds` для каждого кадра;
- порог `iou_threshold`;
- режим `class_agnostic`:
  - `true` — класс игнорируется,
  - `false` — сравнение только по совпадающему `class_id`.

**Шаги алгоритма (на кадр):**
1. Получение списка GT bbox по ключу `source#frame`.
   - Если GT для кадра нет, все предсказания учитываются как `FP`.
2. Сброс флагов `matched=false` для GT.
3. Для каждого предсказания:
   - поиск **не сопоставленного** GT с максимальным IoU (и совпадающим классом при `class_agnostic=false`);
   - если `best_iou ≥ threshold` -> `TP++`, GT помечается `matched=true`;
   - иначе → `FP++`.
4. После обработки предсказаний:
   - каждый GT с `matched=false` даёт `FN++`.

### 7.10 Формирование результатов: CSV и визуализация

**Запись CSV**  
**Где реализовано:** `CsvWriter` — `src/csv.cpp`  
- При создании файла записывается заголовок.  
- Для каждой детекции сериализуются: источник, номер кадра, `track_id`, `class_id`, имя класса, confidence и bbox `(x,y,w,h)`.

**Визуализация bbox**  
**Где реализовано:** `draw_detections` и `color_for_id` — `src/main.cpp`  
- Цвет выбирается детерминированно из `track_id` (если есть) или `class_id`:
  - hue = `(id * 37) mod 180`, фиксированные sat/val;
  - преобразование HSV→BGR;
  - слишком тёмные цвета заменяются на зелёный.
- Для каждого bbox рисуется прямоугольник и плашка с подписью: `<class_name> [id=...] <conf>`.

### 7.11 Вспомогательные утилиты

**Где реализовано:** `src/utils.cpp`, частично `src/main.cpp`

- Приведение расширений к нижнему регистру для классификации входных файлов (`to_lower`).
- Создание директорий с обработкой ошибок (`ensure_dir`).
- Накопление и форматирование статистики времени (`sum`, `format_float`).
- Перечисление изображений в директории и сортировка по имени (`list_images`).

---

## 8) Обучение YOLO + экспорт в ONNX
Была натренирована модель `yolov8n.pt`

### 8.1 Python окружение
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 8.2 Установка Ultralytics
```bash
pip install ultralytics opencv-python pyyaml
```

### 8.3 Установка PyTorch (рекомендуется с CUDA)
PyTorch устанавливается по официальной команде с сайта PyTorch под среду.
После установки проверьте:
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

### 8.4 Конвертация COCO -> YOLO
Использованный датасет Seadronessee анотирован в формате COCO. YOLO же ожидает одноименную нотацию.
Для конвертации используются следующие написанные утилиты:
```bash
python tools/convert_coco_to_yolo.py \
  --coco_json=datasets/seadronessee_raw/annotations/instances_train.json \
  --images_dir=datasets/seadronessee_raw/images/train \
  --out_dir=datasets/seadronessee_yolo/train

python tools/convert_coco_to_yolo.py \
  --coco_json=datasets/seadronessee_raw/annotations/instances_val.json \
  --images_dir=datasets/seadronessee_raw/images/val \
  --out_dir=datasets/seadronessee_yolo/val
```

Должно получиться:
```
datasets/seadronessee_yolo/train/images
datasets/seadronessee_yolo/train/labels
datasets/seadronessee_yolo/train/classes.txt

datasets/seadronessee_yolo/val/images
datasets/seadronessee_yolo/val/labels
datasets/seadronessee_yolo/val/classes.txt
```

Необходимо скопировать любой файл `classes.txt` в папку `models` (они должны быть идентичны).
```
models/classes.txt
```

### 8.5 dataset.yaml
Пример имён классов (в правильном порядке):
```
swimmer
boat
jetski
life_saving_appliances
buoy
```

Для обучения модели необходимо создать yaml:
```bash
python tools/make_dataset_yaml.py \
  --train_dir=datasets/seadronessee_yolo/train \
  --val_dir=datasets/seadronessee_yolo/val \
  --names="swimmer,boat,jetski,life_saving_appliances,buoy" \
  --out=datasets/seadronessee_yolo/seadronessee.yaml
```

### 8.6 Обучение
```bash
yolo detect train \
  data=datasets/seadronessee_yolo/seadronessee.yaml \
  model=yolov8n.pt \
  imgsz=640 \
  epochs=50 \
  batch=16 \
  device=0
```

### 8.7 Экспорт в ONNX
```bash
yolo export \
  model=runs/detect/train/weights/best.pt \
  format=onnx \
  imgsz=640
```

Перенесите результат:
```
models/best.onnx
```

---

## 9) GT CSV для метрик

```bash
python tools/yolo_to_gt_csv.py \
  --images_dir=datasets/seadronessee_yolo/val/images \
  --labels_dir=datasets/seadronessee_yolo/val/labels \
  --out=gt_val.csv
```

---

## 10) Бенчмарк
```bash
./build/seasurface_detector \
  --input=datasets/seadronessee_yolo/val/images \
  --output=out_eval \
  --csv=out_eval/dets.csv \
  --gt_csv=gt_val.csv \
  --eval=true \
  --iou=0.5 \
  --class_agnostic=true \
  --model=models/best.onnx \
  --classes=models/classes.txt \
  --use_gpu=true
```

## 11) Выводы
Резудьтаты бенчмарка (разрешение `1230x933`, `IoU=0.5`, `1547` кадров) следующие:
- Общее время обработки: `71.933 с`
- Среднее время обработки кадра: `46.498 мс`
- `TP: 6261`
- `FP: 1157`
- `FN: 3369`
- `Precision: 0.8440`
- `Recall: 0.6502`

Более лучших результатов можно добиться, если:
- Тренировать более тяжелую модель (`yolov8n.pt` -- самая маленькая)
- Тренировать на большем разрешении (использовалось `640x640`)
- При конвертации датасета отбросить лишние классы (например, `swimmer`)
- Использовать другие параметры `conf`, `nms`, `iou`, `max_missed`