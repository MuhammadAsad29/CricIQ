# 🏏 CricIQ — Ball-by-Ball Match Outcome Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Gradio-UI-orange?logo=gradio&logoColor=white" />
  <img src="https://img.shields.io/badge/T20%20Cricket-Predictor-green" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

> **CricIQ** is a deep learning project that uses an LSTM neural network trained on **3,723 T20 cricket matches** to predict — ball by ball — the win probability, final score, and next-ball outcome for any live match scenario.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🔍 Overview

CricIQ is trained on ball-by-ball data from three major T20 datasets:

- **ICC Men's T20 World Cup**
- **Pakistan Super League (PSL)**
- **All international T20 male matches**

The model processes each match as a **time-series sequence of 120 balls** and simultaneously predicts:

| Output Head               | Task                                          | Type                   |
| ------------------------- | --------------------------------------------- | ---------------------- |
| **Win Probability** | Will the batting team win?                    | Binary Classification  |
| **Final Score**     | What will the innings total be?               | Regression             |
| **Next Ball**       | Dot / Boundary / Wicket on the next delivery? | 3-class Classification |

---

## 🎮 Demo

The project ships with an interactive **Gradio web UI** that runs locally in your browser:

```
🏏 CricketIQ — Ball-by-Ball Match Outcome Predictor
```

Fill in the current match situation and click **Predict** to get:

- 📣 Live commentary on the game state
- 📋 Full match snapshot (score, overs, required rate)
- 🏆 Win probability bars for both teams
- 🎯 Predicted innings final score
- ⚡ Next ball probabilities (Dot / Boundary / Wicket)

---

## ✨ Features

- **825,793 ball-by-ball rows** parsed from 3,723 JSON match files
- **Multi-task LSTM** with 3 output heads trained simultaneously
- **Feature engineering** — current run rate, required run rate, balls remaining, innings context
- **Toss adjustment** — ±3% soft bias to win probability based on toss result
- **Interactive Gradio UI** — no coding required to run predictions
- **Trained model checkpoint** (`cricketiq_best.pt`) included for immediate use

---

## 📂 Dataset

Data sourced from [Cricsheet](https://cricsheet.org/) in JSON format:

| File                                | Description                   | Matches |
| ----------------------------------- | ----------------------------- | ------- |
| `icc_mens_t20_world_cup_json.zip` | ICC T20 World Cup matches     | ~500+   |
| `psl_json.zip`                    | Pakistan Super League matches | ~700+   |
| `t20s_male_json.zip`              | All international male T20s   | ~2500+  |

> **Total:** 3,723 matches → 825,793 ball-by-ball rows → 7,205 innings sequences

### Features per Ball (9 total)

| Feature             | Description                                         |
| ------------------- | --------------------------------------------------- |
| `ball_number`     | Ball index within the innings (0–119), normalized  |
| `runs_so_far`     | Cumulative runs scored, normalized                  |
| `wickets_so_far`  | Wickets fallen so far, normalized                   |
| `balls_remaining` | Balls left in the innings, normalized               |
| `run_rate`        | Current run rate (runs/ball), clipped & normalized  |
| `target_score`    | 1st innings total (2nd innings only), normalized    |
| `runs_needed`     | Runs required to win (2nd innings only), normalized |
| `req_run_rate`    | Required run rate (2nd innings only), normalized    |
| `inning`          | 0 = 1st innings, 1 = 2nd innings                    |

---

## 🧠 Model Architecture

```
CricketIQ_LSTM
├── LSTM Backbone (shared)
│   ├── Input:  [batch, 120 balls, 9 features]
│   ├── Layers: 2 stacked LSTM layers
│   ├── Hidden: 128 units per layer
│   └── Dropout: 0.3
│
├── Head 1 — Win Probability
│   ├── Input:  Last hidden state [batch, 128]
│   └── Output: Sigmoid → win probability [batch]
│
├── Head 2 — Final Score Regression
│   ├── Input:  Last hidden state [batch, 128]
│   └── Output: Sigmoid → normalized score [batch]
│
└── Head 3 — Next Ball Classification
    ├── Input:  All time steps [batch, 120, 128]
    └── Output: Softmax → 3-class probabilities [batch, 120, 3]
```

**Total parameters:** ~228,357

### Loss Functions

| Head            | Loss                  | Weight |
| --------------- | --------------------- | ------ |
| Win probability | `BCEWithLogitsLoss` | 1.0    |
| Final score     | `MSELoss`           | 0.5    |
| Next ball       | `CrossEntropyLoss`  | 0.5    |

**Optimizer:** Adam (lr=1e-3) with `ReduceLROnPlateau` scheduler

---

## 📁 Project Structure

```
CricIQ/
├── CricIQ.ipynb                       # Main notebook (data → model → UI)
├── cricketiq_best.pt                  # Best trained model checkpoint
├── requirements.txt                   # Python dependencies
├── icc_mens_t20_world_cup_json.zip    # ICC T20 World Cup dataset
├── psl_json.zip                       # PSL dataset
└── t20s_male_json.zip                 # International T20 dataset
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CricIQ.git
cd CricIQ
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install gradio
```

> **GPU Note:** If you have a CUDA-capable GPU, PyTorch will automatically use it. The model also runs fine on CPU.

---

## 🚀 Usage

### Option A — Run the full notebook

Open `CricIQ.ipynb` in Jupyter or VS Code and run all cells in order:

| Cell             | Description                                       |
| ---------------- | ------------------------------------------------- |
| **Cell 1** | Unzip datasets & parse 825K ball-by-ball rows     |
| **Cell 2** | Feature engineering & sequence creation           |
| **Cell 3** | Define LSTM model architecture                    |
| **Cell 4** | Train/Val/Test split & training loop (20 epochs)  |
| **Cell 5** | Manual inference — type a match scenario in code |
| **Cell 6** | Launch the Gradio interactive web UI              |

### Option B — Skip training, use pre-trained model

The trained checkpoint `cricketiq_best.pt` is included. Jump directly to **Cell 5** or **Cell 6** to run inference without retraining.

### Option C — Gradio UI

Run Cell 6 to launch the web app at `http://127.0.0.1:7860`:

```
Runs Scored So Far         : [  100  ]
Wickets Fallen             : [   2   ]
Balls Bowled (0–120)       : [  80   ]
Target Score               : [  160  ]
Innings                    : 2nd Innings ●
Did Batting Team Win Toss? : Yes ●

                       [ 🔮 Predict ]
```

---

## 📊 Results

### Training Summary (20 Epochs)

| Split          | Size                  | Win Accuracy    |
| -------------- | --------------------- | --------------- |
| Train          | 5,403 innings         | ~82.6%          |
| Validation     | 1,081 innings         | ~83.2%          |
| **Test** | **721 innings** | **84.2%** |

### Best Model

- **Best Validation Loss:** 0.6248
- **Test Loss:** 0.6225
- **Test Win Accuracy: 84.19%**

### Next-Ball Class Distribution

| Class | Label             | Count   |
| ----- | ----------------- | ------- |
| 0     | Dot / Other       | 709,189 |
| 1     | Boundary (4 or 6) | 111,683 |
| 2     | Wicket            | 43,728  |

---

## ⚙️ How It Works

```
Raw JSON Match Files
        │
        ▼
  parse_match()          ← Ball-by-ball parsing
        │
        ▼
  Feature Engineering    ← run_rate, req_run_rate, balls_remaining …
        │
        ▼
  Sequence Builder       ← [N innings × 120 balls × 9 features]
        │
        ▼
  CricketIQ_LSTM         ← 2-layer LSTM, 128 hidden units
        │
   ┌────┴──────┬──────────┐
   ▼           ▼          ▼
Win Head   Score Head  Next-Ball Head
(binary)  (regression) (3-class / ball)
```

At inference, you provide the **current match state** (runs, wickets, balls bowled, innings, target) and the model returns all three predictions **instantly**.

---

## 🛠️ Tech Stack

| Library      | Version | Purpose               |
| ------------ | ------- | --------------------- |
| Python       | 3.10+   | Core language         |
| PyTorch      | 2.12.0  | LSTM model & training |
| NumPy        | 2.4.6   | Numerical operations  |
| Pandas       | 3.0.3   | Data manipulation     |
| Gradio       | latest  | Interactive web UI    |
| scikit-learn | —      | Train/val/test split  |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Ball-by-ball data from [Cricsheet](https://cricsheet.org/) — an incredible open cricket data resource
- UI built with [Gradio](https://gradio.app/)
- Model training accelerated on Google Colab (NVIDIA T4 GPU)

---

<p align="center">Made with ❤️ and cricket passion 🏏</p>
