import torch
import cv2
import mediapipe as mp
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

from config import BEST_MODEL_DIR


def real_time_detection(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageClassification.from_pretrained(model_path).to(device)
    processor = AutoImageProcessor.from_pretrained(model_path)

    id2label = model.config.id2label
    class_labels = [id2label[i] for i in range(len(id2label))]

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image_rgb.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

            face_image = image_rgb[y:y+h, x:x+w]
            if face_image.size == 0:
                continue

            face_pil = Image.fromarray(face_image).resize((224, 224), Image.LANCZOS)

            inputs = processor(face_pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()

            top3_probs, top3_indices = torch.topk(probabilities, 3)
            prob_text = "\n".join([f"{class_labels[idx]}: {prob * 100:.2f}%" for idx, prob in zip(top3_indices.cpu().numpy(), top3_probs.cpu().numpy())])

            image_with_box = frame.copy()
            cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_with_box, class_labels[predicted_class], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(image_with_box, prob_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            image_with_box = frame.copy()
            cv2.putText(image_with_box, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Facial Expression Recognition', image_with_box)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()


if __name__ == "__main__":
    real_time_detection(BEST_MODEL_DIR)