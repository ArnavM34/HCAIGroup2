import os
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

print("🔍 Step 1: Starting batch prediction script...")

# === Setup Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'resnet50_food101.pth')
IMAGES_DIR = os.path.join(BASE_DIR, 'food_photos')  # folder containing images to predict
print(f"📁 Step 2: BASE_DIR set to {BASE_DIR}")
print(f"📁 Step 3: Checking model path: {MODEL_PATH}")
print(f"📁 Step 4: Checking images directory: {IMAGES_DIR}")

# === Check model and images directory exist ===
if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: Model file not found at:", MODEL_PATH)
    exit()
print("✅ Step 5: Model file found.")

if not os.path.exists(IMAGES_DIR):
    print("❌ ERROR: Images directory not found at:", IMAGES_DIR)
    exit()
print("✅ Step 6: Images directory found.")

# List all image files in IMAGES_DIR (simple filter for jpg/png)
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    print("❌ ERROR: No images found in folder.")
    exit()
print(f"✅ Step 7: Found {len(image_files)} image(s) to predict.")

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Step 8: Using device: {device}")

# === Define Image Transforms ===
print("🔧 Step 9: Setting up image transforms...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
print("✅ Step 10: Transforms ready.")

# === Load Model ===
print("📦 Step 11: Loading ResNet50 model...")
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 101)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)
print("✅ Step 12: Model loaded and ready.")

# === Load Class Names ===
class_names_path = os.path.join(BASE_DIR, 'train')
if not os.path.exists(class_names_path):
    print("❌ ERROR: 'train' folder not found.")
    exit()
class_names = sorted(os.listdir(class_names_path))

# === Predict Each Image ===
print("🍽️ Step 13: Starting predictions for each image:")
for img_file in image_files:
    img_path = os.path.join(IMAGES_DIR, img_file)
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_catid = torch.max(probabilities, 1)
        predicted_label = class_names[top_catid.item()]
        print(f" - {img_file}: {predicted_label} (score: {top_prob.item():.4f})")
    except Exception as e:
        print(f"❌ Failed to process {img_file}: {e}")
