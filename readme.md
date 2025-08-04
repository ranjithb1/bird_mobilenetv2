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


