# plant-disease-detection
Plant Disease Detection project

# 🌿 GreenLeaf: A Web Application for AI-Based Plant Disease Detection with Chat and Visual Analysis

GreenLeaf is a containerized AI-powered web application designed to detect plant diseases from leaf images using a CNN model. It offers an interactive chat-based interface for guidance, educational insights, and visual analysis. Built with Flask, PyTorch, and Hugging Face integrations, it aims to empower farmers, researchers, and agriculture enthusiasts.

---

## 🚀 Features

- **🌱 Disease Detection**: Predicts plant diseases through a pre-trained CNN model.
- **💬 Chatbot Assistant**: Provides instant guidance, disease summaries, and plant care tips using offline responses.
- **🖼️ Visual Analysis**: Offers real-time results and stores images for future reference.
- **📁 Containerized**: Fully Dockerized for easy local deployment or cloud hosting.
- **📊 Supplementary Data**: CSV-based info on diseases and recommendations.

## use
- Upload a plant image and detect diseases using a trained model
- Recommend fertilizers based on detected disease
- Simple UI for users
- Dockerized setup
---

## 📂 Repository Structure

| File/Folder       | Description                                         |
|-------------------|-----------------------------------------------------|
| `app.py`          | Main Flask application with routing and prediction |
| `CNN.py`          | Pre-trained PyTorch model for image classification |
| `templates/`      | HTML templates (Jinja2) for the web interface       |
| `static/`         | Uploaded images and assets                         |
| `disease_info.csv`| Supplementary disease information                  |
| `Dockerfile`      | Docker container configuration                      |
| `.dockerignore`   | Files excluded during Docker build                  |
| `requirements.txt`| Project dependencies                                |

---

## 🧪 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/navaneeta30/greenleaf-ai-plant-disease-detector.git
cd greenleaf-ai-plant-disease-detector

**### 2. Build and Run with Docker**
```bash
docker build -t greenleaf-app .
docker run -p 5000:5000 greenleaf-app

**🧠 Usage Guide**

Upload Leaf Image
Visit the homepage, upload an image of a diseased plant leaf.

Get Prediction
The model classifies the image and shows the disease name.

Chatbot Interaction
Use the chatbot to ask questions about the disease or care tips (offline response simulation).

Reference Information
Disease descriptions and possible treatments are available in disease_info.csv.

**## 🧠 Technologies Used**

- Python
- Flask
- PyTorch 
- OpenCV
- huggingface hub
- Docker
- HTML/CSS (for frontend)
**
## 📈 Future Enhancements**

Extend disease classification to more crops

Mobile-responsive design

Cloud storage integration (e.g., S3 or Firebase)

IoT-based leaf sensors

Crop-specific fertilizer optimization using AI

## 📞 Contact
Feel free to reach out for questions, suggestions, or collaborations:

LinkedIn: Navaneeta
Email: navaneeta3005@gmail.com
