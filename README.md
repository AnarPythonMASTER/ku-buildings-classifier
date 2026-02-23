Here is a professional, beautifully formatted `README.md` file based exactly on the specifications you provided. It tells the story of your project, explains your architectural decisions (like switching to Adam and consolidating classes), and gives clear instructions for anyone who visits your repo.

You can copy everything inside the code block below and paste it directly into your `README.md` file:

```markdown
# KU Ingolstadt Building Classifier 🏫

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![Keras](https://img.shields.io/badge/Keras-3.13.2-red)
![License](https://img.shields.io/badge/Code_License-MIT-green)
![Data License](https://img.shields.io/badge/Data_License-CC_BY_4.0-lightgrey)

## 📌 Project Overview
This project is a custom Convolutional Neural Network (CNN) built to classify images of famous buildings located near the Catholic University of Eichstätt-Ingolstadt (KU) in Bayern, Germany. 

The primary motivation for this project was to apply concepts learned in our **Deep Learning** course taught by Professor Dr. Felix Voigtlaender and Dr. Hannes Matt. Our long-term vision is to simulate a mobile platform where users can scan a building with their phone camera to instantly retrieve its name and information.

## 💾 The Dataset (Custom Data Collection)
**Note:** Due to the large size of the dataset (17+ GB), the raw images and videos are hosted externally on Kaggle.
* 🔗 **Download the dataset here:** [INSERT KAGGLE LINK HERE]

**How we built the dataset:**
* We manually collected over **2,670 raw images** around the MIDS building and KU campus.
* To make the model robust for real-world application, we mimicked the behavior of a person walking along standard pathways and scanning buildings with their phone from various angles and distances.
* We further expanded our dataset by recording walking videos and programmatically extracting frames (2 frames per second) to maximize our training data.

### 🏛️ Target Classes (9 Categories)
After an initial phase where we separated buildings by specific angles (21 classes), we refined our approach to a robust 9-class system representing core structures:
1. Georgianum
2. Entrance door of the Georgianum
3. Seminar room (201-203) in the Georgianum
4. Basement room in the Georgianum
5. The pink building in front of the Georgianum
6. The church on the way to the Mensa
7. The Kreuztor
8. The main entrance of the KU (x80 bus-stop)
9. The Main Building (Hauptbau) / WFI building views

## 🧠 Model Architecture & Training
We built a custom CNN using Keras `Sequential` API. 
* **Input Shape:** 224 x 168 x 3 (RGB images)
* **Architecture:** 3 Convolutional Blocks (32 -> 64 -> 128 filters) with Max Pooling, followed by a Flatten layer, a Dense layer of 64 neurons, and a final Softmax output.
* **Optimization Strategy:** We initially tested SGD but observed poor convergence. Switching to the **Adam Optimizer** with a low learning rate ($10^{-5}$) yielded stable and superior results.
* **Callbacks:** We utilized `EarlyStopping` (restoring best weights), `ReduceLROnPlateau`, and custom `ModelCheckpoint` callbacks to monitor validation loss.

### 📊 Results
* **Validation Accuracy:** ~98%
* **Unseen Test Accuracy:** ~60% (Tested on a completely separate hold-out set of difficult, real-world images not seen during training).

## 📂 Repository Structure
```text
📦 DL-Project
 ┣ 📂 data_raw/          # (Ignored) Original 9 category folders
 ┣ 📂 dataset/           # (Ignored) 80/20 Stratified Train/Val split
 ┣ 📂 logs/              # Training history and epoch metrics
 ┣ 📂 models/            # Saved .keras models and class_mapping.json
 ┣ 📂 test_images/       # Completely unseen real-world test images
 ┣ 📂 videos/            # (Ignored) Source videos for frame extraction
 ┣ 📜 CNN_PROJECT_INGOLSTADT_phase1.ipynb # Phase 1: 21-class architecture
 ┣ 📜 cnn.ipynb          # Phase 2: Final 9-class architecture & training
 ┣ 📜 image_extractor_from_videos.ipynb   # Frame extraction pipeline
 ┣ 📜 split_data.ipynb   # 80/20 stratified data splitting script
 ┣ 📜 implementation.py  # Real-time inference script for new images
 ┣ 📜 requirements.txt   # Python dependencies
 ┗ 📜 README.md

```

## 🚀 How to Run Locally

**1. Clone the repository**

```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName

```

**2. Install Dependencies**

```bash
pip install -r requirements.txt

```

**3. Setup the Data**
Download the dataset from Kaggle and extract it so that your `data_raw` and `dataset` folders are populated.

**4. Run Inference on a New Image**
You can use our pre-trained model to classify your own photos! Ensure your paths inside `implementation.py` point to your local directories, then run:

```bash
python implementation.py

```

*(The script will load `cnn_model_increased_image.keras` and `class_mapping.json` to process images from the `test_images` directory).*

## 👥 Contributors

* **Shahbaz Khalilli** * **Anar Jafarli** ---
*Code licensed under MIT. Dataset licensed under CC BY 4.0.*

```

### Next Steps:
1. Copy the text block above.
2. Open the `README.md` file in your code editor.
3. Paste the text and save the file.
4. Run `git add README.md`, then `git commit -m "Add detailed README"`, and `git push` to update GitHub!

How does the story of your project look to you?

```