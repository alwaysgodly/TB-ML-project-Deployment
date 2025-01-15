from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from cxr_model import CXR_EfficientNetModel 


app = Flask(__name__, template_folder=r'C:\Users\Pushkraj\OneDrive\Desktop\final 3\templates', static_folder=r'C:\Users\Pushkraj\OneDrive\Desktop\final 3\static')

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
model = CXR_EfficientNetModel(num_classes=num_classes).to(device)

# Load the model weights from model(1).pth
model.load_state_dict(torch.load(r"C:\Users\Pushkraj\OneDrive\Desktop\final 3\model\model (1).pth", map_location=device))

model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match EfficientNet input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    try:
        # Preprocess the image
        image = Image.open(file)
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

        # Map prediction to class label
        labels = ["Negative", "Positive"] 
        result = labels[predicted_class.item()]

        return render_template("result.html", prediction=result)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
