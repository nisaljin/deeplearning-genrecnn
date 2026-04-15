import io
import json
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torch import nn

app = FastAPI(title="Multi-Genre CNN Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the dynamic classes from your JSON file
try:
    with open("fma_classes.json", "r") as f:
        metadata = json.load(f)
        CLASSES = metadata["classes"]
        NUM_CLASSES = metadata["num_classes"]
except FileNotFoundError:
    raise RuntimeError("Could not find fma_classes.json! Please ensure it is in the root directory.")

# 2. The Updated Multi-Label Architecture
class MultiGenreCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)) # Handles dynamic audio lengths!
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# 3. Load the Model
model = MultiGenreCNN(NUM_CLASSES).to(device)
try:
    # Use map_location to ensure it loads on a Mac/CPU even if trained on a GPU
    model.load_state_dict(torch.load("outputs/best_fma_multilabel.pt", map_location=device))
    model.eval()
    print("✅ Multi-Label Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model weights: {e}")

# Global Audio Transformers
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512
).to(device)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "multi-label", "classes_loaded": NUM_CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int | None = 5) -> dict:
    filename = file.filename or ""
    if not filename.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a", ".webm", ".weba", ".mp4")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        audio_bytes = await file.read()
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

        # Audio preprocessing
        if sr != 22050:
            waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Standardize length to 5 seconds
        target_length = 22050 * 5
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))

        waveform = waveform.to(device)
        mel_spec = amplitude_to_db(mel_transform(waveform)).unsqueeze(0)

        # 4. MULTI-LABEL PREDICTION LOGIC
        with torch.no_grad():
            output = model(mel_spec)
            probabilities = torch.sigmoid(output)[0]

        results = []
        
        # Grab ALL genres and their raw probabilities
        for i, prob in enumerate(probabilities):
            results.append({
                "genre": CLASSES[i],
                "probability": prob.item()
            })

        # Sort them by highest confidence first
        results.sort(key=lambda x: x["probability"], reverse=True)

        # Always return the top_k (usually 5) predictions so the UI always has multiple bars!
        return {"predictions": results[:top_k]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("infer_api:app", host="0.0.0.0", port=8000, reload=True)