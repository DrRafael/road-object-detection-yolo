import os
from typing import Dict, List, Any
from imageai.Detection import ObjectDetection


ROAD_CLASSES = {
    "car", "motorbike", "bicycle", "person", 
    "bus", "train", "truck", "traffic light", "stop sign"
}


def detect_objects_on_road(
    input_image_path: str, 
    output_image_path: str, 
    model_path: str, 
    min_probability: int = 30
) -> List[Dict[str, Any]]:
    """Detects objects in an input image using a pre-trained YOLOv3 model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLOv3 model weights not found at: {model_path}")
    
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found at: {input_image_path}")

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()

    detections = detector.detectObjectsFromImage(
        input_image=input_image_path,
        output_image_path=output_image_path,
        minimum_percentage_probability=min_probability
    )

    return detections


def analyze_road_objects(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filters detected objects to isolate relevant road traffic participants and infrastructure."""
    return [
        detection for detection in detections 
        if detection.get("name") in ROAD_CLASSES
    ]


def display_road_safety_rules() -> None:
    """Displays informative road safety awareness guidelines."""
    print("\n--- SafetyAI Road Safety Awareness ---")
    print("1. Always obey traffic lights and road signs.")
    print("2. Utilize designated pedestrian crossings.")
    print("3. Avoid jaywalking and maintain spatial awareness on roadways.")
    print("4. Remain predictable and safe in traffic environments.\n")


def main():
    input_image = "image.jpg"
    output_image = "output_image.jpg"
    model_path = "yolov3.pt"

    try:
        raw_detections = detect_objects_on_road(input_image, output_image, model_path)
        road_objects = analyze_road_objects(raw_detections)

        if road_objects:
            print(f"Detected {len(road_objects)} road participant(s):")
            for obj in road_objects:
                print(f" - {obj['name']}: {obj['percentage_probability']:.2f}% | Box: {obj['box_points']}")
        else:
            print("No traffic participants detected in the provided image.")

        display_road_safety_rules()

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")


if __name__ == "__main__":
    main()
