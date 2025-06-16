import ImageAI
from imageai.Detection import ObjectDetection


def detect_objects_on_road(input_image, output_image, model_path):
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()

    detections = detector.detectObjectsFromImage(
        input_image=input_image,
        output_image_path=output_image,
        minimum_percentage_probability=30
    )

    return detections

def analyze_objects(detections):
    road_objects = []
    if len(detections) > 0:
      for detection in detections:
          if detection["name"] in ["car", "motorbike", "bicycle", "person", "bus", 'train', 'truck','traffic_light', 'stop_sign']:
              road_objects.append(detection)

    return road_objects

def road_safety_rules():
    print()
    print("Привет! Это SafetyAI - приложение для безопасности на дороге.")
    print("Правила безопасности на дороге очень важны, и я помогу вам их запомнить.")
    print("Помните, что всегда соблюдайте правила дорожного движения и будьте внимательны на дороге.")
    print("Пользуйтесь светофорами и пешеходными переходами.")
    print("Никогда не переходите дорогу в неположенном месте.")
    print("И помните, что на дороге всегда нужно быть осторожным и предсказуемым.")
    print("Будьте внимательны на дороге и удачи!")

input_image = "image.jpg"
output_image = "output_image.jpg"

detections = detect_objects_on_road(input_image, output_image, "/content/yolov3.pt")
road_objects = analyze_objects(detections)

if len(road_objects) > 0:
  print("Обнаруженные участники дорожного движения:")
  for obj in road_objects:
      print(obj["name"], " : ", obj["percentage_probability"], " : ", obj["box_points"])
else:
   print("Ни одного участника дорожного движения не обнаружено!")

road_safety_rules()
