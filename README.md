# 🧠 ML Studio — Teachable Machine Klon

> **Türkmen / English** · Bilingual AI öwretmek platformasy

---

## 📌 Taslama Barada | About the Project

**TM:** ML Studio — Google Teachable Machine-a meňzeş, emeli intellekti görkezme we öwretmek üçin niýetlenen web platformasy. Flask + scikit-learn arkaly hakyky ML modelleri işleýär.

**EN:** ML Studio is a Google Teachable Machine-style web platform for interactive machine learning. It runs real ML models via Flask + scikit-learn backend.

---

## ✨ Esasy Funksiýalar | Key Features

| Funksiýa | EN |
|---|---|
| 📷 Webcam bilen nusga almak | Webcam sample capture |
| 🖼 Surat ýükläp synag etmek | Image upload & live test |
| 🤖 Gözegçilikli öwrenmek | Supervised Learning |
| 🔍 Gözegçiliksiz öwrenmek | Unsupervised / Clustering |
| 🎮 Güýçlendirme öwrenmesi | Reinforcement Learning (Q-Learning) |
| 🌐 Türkmençe / Iňlisçe UI | Bilingual UI (TM / EN) |
| 📤 Export / Share | Model export & share modal |

---

## 🗂 Faýl Gurluşy | File Structure

```
ml_studio/
├── app.py              # Flask backend (API)
├── templates/
│   └── index.html      # Frontend (Teachable Machine UI)
├── requirements.txt    # Python dependencies
└── README.md           # Bu faýl / This file
```

---

## ⚙️ Gurnalyş | Installation

### 1. Talaplar | Requirements

- Python 3.9+
- pip

### 2. Gurnamak | Install dependencies

```bash
pip install flask scikit-learn numpy
```

ýa-da / or:

```bash
pip install -r requirements.txt
```

### 3. Işletmek | Run

```bash
# Faýl gurluşyny düz / Set up file structure:
mkdir -p ml_studio/templates
cp app.py ml_studio/
cp index.html ml_studio/templates/

# Işlet / Run:
cd ml_studio
python app.py
```

### 4. Açmak | Open

```
http://localhost:5050
```

---

## 🔌 API Endpoint-ler | API Endpoints

| Method | Endpoint | Beýany / Description |
|--------|----------|----------------------|
| `POST` | `/api/supervised/generate` | Dataset döret / Generate dataset |
| `POST` | `/api/supervised/train` | Modeli türgenleşdir / Train model |
| `POST` | `/api/unsupervised/generate` | Klaster datasy döret |
| `POST` | `/api/unsupervised/cluster` | Klasterleşdirmegi işlet |
| `POST` | `/api/rl/train` | Q-Learning agenti türgenleşdir |
| `GET`  | `/` | Baş sahypa / Main page |

---

## 🧩 Modullar | Modules

### 1. 🤖 Gözegçilikli Öwrenmek (Supervised Learning)

- **Algoritmlar:** KNN, Decision Tree, Logistic Regression, SVM
- **Webcam** bilen hakyky surat nusgasy almak
- **Upload** bilen köp surat ýüklemek
- Karar serhet (Decision Boundary) görkezme
- Klass-laýyk Precision, Recall, Support tablisasy
- Ýitgi egri çyzgy (Training Loss Curve)
- **Janly Synag:** webcam ýa-da faýl arkaly hakyky çaklama

### 2. 🔍 Gözegçiliksiz Öwrenmek (Unsupervised Learning)

- **Algoritmlar:** K-Means, DBSCAN, Hierarchical, Gaussian Mixture
- Maglumatlary dörediň ýa-da CSV ýükläň
- Silhouette Score, Inertia görkezijileri
- Klaster reňkli scatter plot
- Konwergensiýa grafigi

### 3. 🎮 Güýçlendirme Öwrenmesi (Reinforcement Learning)

- **5×5 Grid World** gurşawy
- Tabular **Q-Learning** algoritmi
- Öýjüklere basmak bilen diwar/howp goşmak
- α (learning rate), γ (discount), ε (exploration) sazlama
- Öwrenilen syýasat (policy) ok-nyşanlar bilen görkezilýär
- Agenti görsel ýörediş animasiýasy

---

## 📷 Webcam Ulanylyşy | Webcam Usage

**Nusga almak üçin:**
1. Supervised bölümünde class kartynyň **📷 Webcam** düwmesine basyň
2. Modal açylýar → **Capture** bilen bir surat alyň
3. ýa-da **⏺ Hold to Record** düwmesini tutup awtomatik surat alyň (5/sek)
4. **Done** bilen ýapyň

**Preview panelinde Janly Synag:**
- **Webcam tab** → kamerany açyň → awtomatik çaklama 1.2s-de bir gezek
- **Upload tab** → suraty drag-drop ýa-da saýlaň → derrew çaklama

---

## 🔗 Paýlaşmak | Share & Export

- **Share** düwmesi → link göçürme, Telegram / WhatsApp / Email
- **Export Model** → `model.json` faýl göçürip almak
- JSON faýlda: algoritm ady, klass atlary, takyklyk, timestamp

---

## 🌐 Dil Geçişi | Language Switch

Sag ýokarda **EN | TM** düwmeleri bilen geçiň. Saýlan dil `localStorage`-da ýatda galýar.

Switch using **EN | TM** buttons in the top-right. Language preference is saved in `localStorage`.

---

## 🛠 Tehniki Stack | Tech Stack

| Gatlag | Tehnologiýa |
|--------|-------------|
| Backend | Python 3.9+, Flask 2.x |
| ML | scikit-learn (KNN, SVM, KMeans, DBSCAN, GMM) |
| Matematika | NumPy |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Şriftler | Google Fonts (Google Sans, Roboto Mono) |
| Grafika | Canvas API |
| Kamera | `getUserMedia` Web API |

---

## 📋 requirements.txt

```
flask>=2.0
scikit-learn>=1.2
numpy>=1.24
```

---

## 🎓 Diplom Taslama | Diploma Project

Bu taslama **Oguz Han Inžener we Tehnologiýa Uniwersitetiniň** talyby tarapyndan diplom işi hökmünde taýýarlanyldy.

This project was developed as a diploma thesis at **Oguz Han Engineering and Technology University of Turkmenistan**.

---

## 📄 Lisenziýa | License

MIT License — erkin ulanylyş we üýtgetmek üçin açyk.

---

*ML Studio · v1.0 · 2025*