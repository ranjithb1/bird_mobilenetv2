Bird Species Classifier using Transfer Learning (MOBILENETV2)
Comparison of Different popular Image Classfication models, results and taking up suitable high perfomed model for efficiency.
Leverage MobilenetV2 as a feature extractor by adding custom classification layers on top.
Preprocess and augment image data using Keras' ImageDataGenerator to improve model robustness.
Split the dataset into training, validation, and test sets for proper model evaluation.
Train the model with callbacks such as early stopping and learning rate reduction to prevent overfitting.
Evaluate model performance using metrics like accuracy, confusion matrix, and classification report.
Visualize predictions by displaying actual vs predicted classes on random test images.
Save the trained model in .h5 format for reuse or further training.
Quantize and convert the model to .tflite with int16 optimization for faster inference and smaller size, ideal for Android/Web deployment.
🚀 Local Deployment using Flask

After training and quantizing the model, we implemented a lightweight web interface to run the bird species prediction locally using:
Flask – to serve the model and handle image uploads 3 .NumPy – for numerical operations and preprocessing
TensorFlow / Keras – to load the trained model
Pillow (PIL) – to handle image input and resizing
Gunicorn – for running the Flask server in production-ready environments
DEMO

Model Information
Trained model: bird_species.h5
Input size: 224x224 RGB images
Classes: American Goldfinch Barn Owl Carmine Bee-Eater Downy Woodpecker Emperor Penguin Flamingo

✨ Features

Upload any bird image and get instant prediction
Animated background using Vanta.js (Birds)
Elegant UI with glassmorphism and custom CSS
Displays confidence percentage for prediction
📸 How to Use

Clone the repository:
git clone https://github.com/ranjithbq/PROJECT_TEST1.git
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
Install dependencies:

pip install -r requirements.txt
Run the Flask app:
python app/views.py
Open your browser and go to http://127.0.0.1:5000/.
🌐 Deployment

After developing the app locally, we deployed it to a live website using Render:
Pushed the project to a GitHub repository.
Created a new Web Service on Render.
Connected the GitHub repository to Render.
Specified the build command and start command:
Build Command: (leave blank or use pip install -r requirements.txt if needed)
Start Command: python app/views.py
Selected Python as the environment.
Deployed the app.
✅ After successful deployment, the web app is live and accessible on the internet.

✨ You can try it here: https://bird-mobilenetv2-1.onrender.com

📱 Android App Conversion

After deploying the website, we converted it into an Android app using Appilix:
Updated the HTML, CSS, and JavaScript for better Android WebView compatibility:
Ensured responsive design.
Enabled camera/image upload features to work in mobile browsers.
Used Median.co to convert the live website into an Android app:
Entered the deployed website URL.
Customized the app name, icon, and splash screen.
Generated and downloaded the .apk file.
✅ The Android app is now ready to install and use!

📦 Download the APK here: https://median.co/share/xkymxy#apk

🚀 App Workflow and Features

Once the app is opened on an Android device:
The app connects to the Render-hosted backend and loads the web interface.
Users can upload an image in two ways:
By capturing a photo using the camera.
By selecting an existing file using the choose file option.
After uploading, the app:
Shows a preview of the selected image.
Offers a "Predict" button.
Upon clicking Predict: The image is processed by the deployed TensorFlow Lite model.
The app displays: The predicted bird species name.
The confidence score.
inference , preprocesiignt time to predict it 
A link to more information about the predicted bird.
✅ This makes the app fully functional for real-time bird species recognition!



<!DOCTYPE html>

<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bird Species Classifier</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
      background: #f9f9f9;
      color: #333;
      line-height: 1.6;
    }
    header, footer, .content {
      max-width: 1000px;
      margin: auto;
      padding: 20px;
    }
    header {
      text-align: center;
      background-color: #e8f5fe;
      padding: 30px 20px;
      border-bottom: 2px solid #ccc;
    }
    header h1 {
      margin: 10px 0;
      font-size: 2.5em;
    }
    .social-icons img {
      height: 30px;
      margin: 0 10px;
    }
    .tech-icons img {
      height: 40px;
      margin: 0 8px;
    }
    .code-block {
      background: #eee;
      padding: 15px;
      border-radius: 5px;
      font-family: monospace;
      overflow-x: auto;
    }
    .section-title {
      text-align: center;
      margin-top: 50px;
      font-size: 1.5em;
      border-bottom: 1px solid #ccc;
      padding-bottom: 10px;
    }
    .center {
      text-align: center;
    }
    .gif-right {
      float: right;
      height: 230px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Bird Species Classifier using Transfer Learning (MobileNetV2)</h1>
    <img src="https://img.shields.io/badge/BirdNetV2-Deployed-green" alt="Model Status">
    <div class="social-icons">
      <a href="https://www.linkedin.com/in/ranjith-ece">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
      </a>
      <a href="#">
        <img src="https://cdn-icons-png.flaticon.com/512/733/733635.png" alt="Twitter">
      </a>
      <a href="#">
        <img src="https://cdn-icons-png.flaticon.com/512/5968/5968853.png" alt="Dev.to">
      </a>
    </div>
    <h3>🕵️ Ranjith | 🎓 ECE Student | 🔬 AI & Embedded Enthusiast</h3>
    <p><img src="https://visitor-badge.laobi.icu/badge?page_id=ranjith-ece.bird-classifier" alt="Visitor Badge"></p>
    <p><i>✨ Fun Fact: I trained this model faster than my coffee cooled down.</i></p>
  </header>

  <div class="content">
    <img class="gif-right" src="https://media.giphy.com/media/WoD6JZnwap6s8/giphy.gif" alt="Bird GIF">
    <h2 class="section-title">🎓 Project Summary</h2>
    <ul>
      <li>🔄 Compared multiple classification models</li>
      <li>🔗 Used MobileNetV2 for feature extraction</li>
      <li>⚖️ Evaluated with accuracy, confusion matrix, and classification report</li>
      <li>🔢 Converted to TFLite with int16 quantization for deployment</li>
      <li>🏠 Flask web app + Android integration using WebView</li>
    </ul>

```
<h2 class="section-title">🛠️ Tech Stack</h2>
<div class="center tech-icons">
  <img src="https://skillicons.dev/icons?i=python,tensorflow,flask,html,css,js,git,vscode" alt="Tech Stack">
</div>

<h2 class="section-title">📊 Model Details</h2>
<p><b>Trained model:</b> bird_species.h5</p>
<p><b>Classes:</b> American Goldfinch, Barn Owl, Carmine Bee-Eater, Downy Woodpecker, Emperor Penguin, Flamingo</p>

<h2 class="section-title">📅 Quick Start</h2>
<div class="code-block">
  git clone https://github.com/ranjithbq/PROJECT_TEST1.git<br>
  cd PROJECT_TEST1<br>
  python -m venv venv<br>
  source venv/bin/activate  # or venv\Scripts\activate<br>
  pip install -r requirements.txt<br>
  python app/views.py
</div>
<p>Then open: <b>http://127.0.0.1:5000</b></p>

<h2 class="section-title">🌐 Deployment</h2>
<p>Live Demo: <a href="https://bird-mobilenetv2-1.onrender.com">https://bird-mobilenetv2-1.onrender.com</a></p>
<p>Android APK: <a href="https://median.co/share/xkymxy#apk">Download APK</a></p>

<h2 class="section-title">🚀 App Workflow</h2>
<ol>
  <li>Upload bird image (camera/file)</li>
  <li>View preview and predict</li>
  <li>See predicted class with confidence</li>
  <li>Enjoy smooth Web + Android experience</li>
</ol>

<h2 class="section-title">📁 Project Structure</h2>
<pre class="code-block">
```

PROJECT\_TEST1/
├── app/
│   ├── static/
│   ├── templates/
│   └── views.py
├── model/
│   └── bird\_species.h5
├── data/
├── README.md
├── requirements.txt
└── ... </pre>

```
<h2 class="section-title">✨ Let's Connect</h2>
<p class="center">💬 Reach out on <a href="https://www.linkedin.com/in/ranjith-ece">LinkedIn</a> or star this project!</p>

<h2 class="section-title">💪 Support & Contribution</h2>
<p class="center">✨ Issues, PRs, stars ⭐️, and forks are welcome!</p>
<p class="center">💜 Built with passion for birds, code, and clean UI.</p>

<h2 class="center">✨ Keep Building. Keep Innovating. ✨</h2>
```

  </div>

  <footer class="center">
    <img src="https://raw.githubusercontent.com/mayhemantt/mayhemantt/Update/svg/Bottom.svg" alt="Bottom Banner">
  </footer>
</body>
</html>
