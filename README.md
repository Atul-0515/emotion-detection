# Real-Time Emotion Detection System 😊

A deep learning-based emotion detection system that uses your webcam to recognize 7 different emotions in real-time using a CNN trained on the FER2013 dataset.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- **Real-time emotion detection** from webcam feed
- **7 emotion classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Top-3 confidence scores** displayed on video feed
- **Interactive terminal output** with colored emotions and motivational quotes
- **Multi-face detection** support
- **67%+ validation accuracy** on FER2013 dataset

## 🧠 Model Architecture

- Deep CNN with Batch Normalization
- 2.7M trainable parameters
- Input: 48x48 grayscale images
- Architecture: 4 convolutional blocks + 2 dense layers
- Trained for 50+ epochs with learning rate scheduling and early stopping

## 📊 Dataset

Trained on the **FER2013 dataset**:
- 28,000+ training images
- 7,000+ validation images
- Data augmentation applied (rotation, shift, zoom, flip, brightness)
- Class balancing with weighted loss

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Webcam
- Mac/Linux/Windows

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Atul-0515/emotion-detection.git
cd emotion-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the trained model**

- Download `emotion_model.keras` from [Releases](https://github.com/Atul-0515/emotion-detection/releases)
- Place it in the `models/` folder

4. **Run the application**
```bash
python main.py
```

## 🎮 Usage

### Controls
- **SPACE**: Capture and detect emotions
- **C**: Clear terminal output
- **Q**: Quit application

### Output
The system displays:
- Live video feed with face bounding boxes
- Top 3 emotion predictions with confidence scores
- Detailed terminal output with all 7 emotion probabilities
- Random motivational quotes based on detected emotion

## 📁 Project Structure
```
emotion-detection/
│
├── models/
│   └── emotion_model.keras          # Trained model (download separately)
│
├── main.py                           # Main inference script
├── emotion.py                        # Show emotion change in real time
├── notebooks/
│   └── train.ipynb                   # Training notebook (Google Colab)
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Git ignore file
```

## 📦 Dependencies
```
tensorflow==2.20.0
opencv-python==4.12.0.88
numpy==2.2.6
tabulate==0.9.0
colorama==0.4.6
```

## 🏋️ Training Your Own Model

1. Open `train.ipynb` in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially
4. Training takes approximately 45-60 minutes
5. Download the generated `emotion_model.keras` file

## 📈 Performance

- **Validation Accuracy**: 67.6%
- **Training Accuracy**: 72.0%
- **Real-time FPS**: 30+ (depends on hardware)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FER2013 dataset by Kaggle
- Haar Cascade classifier by OpenCV
- Inspired by various emotion detection projects

## 📧 Contact

Project Link: [https://github.com/Atul-0515/emotion-detection](https://github.com/Atul-0515/emotion-detection)

---

⭐ If you found this project helpful, please give it a star!