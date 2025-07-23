# Bachelor-Project: GUI for Object Detection Using WiFi Signals

A GUI for object detection using simulated WiFi-signals. 

## 📚 Table of Contents

- [📘 Project Description](#-project-description)
- [⚙️ System Description](#-system-description)
- [🧰 Prerequisites](#-prerequisites)
- [✨ Features](#-features)
- [🛠️ Tech Stack](#-tech-stack)
- [📦 Installation](#-installation)
- [🔬 Algorithms and Methods](#-algorithms-and-methods)
- [🧱 Directory Structure](#-directory-structure)
- [🎥 Demo](#-demo)
- [📌 Future Work](#-future-work)


## 📘 Project Description

This project is a Flask-based web GUI for visualizing object detection using WiFi signals and Reconfigurable Intelligent Surfaces (RIS). 
It enables users to simulate signal reflections, estimate object positions using MSE, and visualize the results through plots, histograms, and ROC curves. The Target-Absent vs Target-Present Histogram and Simulated
ROC curves are used to evaluate, whether or not, there is an object present at the estimated coordinate
through evaluating the performance of the detector (See **Algorithms and Methods** section). By having the two statistical methods, it essentially helps us compare
the performance of our detection algorithm.

## ⚙️ System Description

The system involves a 2D indoor channel model with a transmitter (TX) with M antennas, an object
with unknown position, located inside a 2D plane and a Reconfigurable Intelligent Surfaces (RIS) with
N antennas, whose phase shifts can be designed to enhance the signal strength and thus the signal
performance. The RIS act as the passive element that reflects the signal to the object of interest with
applied phase shifts, which enables the possibility of detecting changes.
In our project, we only consider a single transmitter and RIS at locations (3,0) and (0,3), respectively
for the sake of simplicity.


## 🧰 Prerequisites

- Python 3.8+ (Python 3.12.1 recommended)
- pip (Python package manager)
- Visual Studio Code (Recommended)


## ✨ Features

- **User Login System** – Simple authentication with predefined credentials  
- **Graphical User Interface** – Built with HTML/CSS and Flask for the server part
- **Signal Reflection Simulation** – Models how WiFi signals interact with objects and RIS  
- **Estimation** – Predicts object location using Mean Squared Error (MSE)  
- **Statistical Visualization** – ROC curve and histograms for evaluating detection reliability


## 🛠️ Tech Stack

- **Backend**: Python, Flask  
- **Frontend**: HTML, CSS  
- **Data & Logic**: Numpy, Matplotlib  
- **Authentication**: Basic session-based login system  
- **Visualization**: Plotting with Matplotlib (saved and embedded in UI)


## 📦 Installation

```bash
git clone https://github.com/yourusername/GUI-for-Object-Detection-using-WiFi-Signals.git
cd GUI-for-Object-Detection-using-WiFi-Signals
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then go to http://127.0.0.1:5000/login and use the credentials: 

```bash
username: Brian123
password: JamesJac123
```

## 🔬 Algorithms and Methods

- **Channel Modeling:** Simulates the propagation of WiFi signals reflecting off a Reconfigurable Intelligent Surface (RIS) to reach a target object.
- **MSE Estimation:** Object position is estimated by minimizing the Mean Squared Error between simulated and expected signals.
- **Statistical Detection:** Uses probabilistic distributions via histograms to distinguish between the presence and absence of a target. It tells us, how accurate the detector separates between when an object is present vs when an object is absent.
- **ROC Curve Analysis:** Provides insight into the trade-off between false alarms and detection accuracy. With ROC-curve, it tells us, how well the detector can detect objects, while also reject incorrect signal
values, when an object is absent.

## 🧱 Directory Structure 

```

├── Project.py             # Flask application logic
├── Algorithm.py           # Room layout and plot generation
├── Algorithm_RIS.py       # RIS-enhanced channel simulation & estimation
├── al.py                  # Statistical modeling and ROC computation
├── styles.css     # Frontend styling
├── templates/
├── login.html     # Login page
├── New_Page.html  # Main page
├── about.html     
├── support.html
└── README.md

```

## 🎥 Demo

## 📌 Future Work

- Tracking objects when moving for true and estimated localization would be a nice-to-have feature.
  
- More secure login with storing credentials in database as shown in the **📦 Installation** section. As an extension, whenever fan/route is stored on "history" part on web browser,
  the user can basically cheat by changing the route to the desired route location, and the user can access the page.
  If history of the route is cleared from web browser, the user has to login with correct credentials, otherwise login is not possible.

- Designing a new algorithm, where we include RIS (ris set to 0) to enhance the signal.
  
- Register users safely in database, since in current project, we store it manually in python web framework Flask, which is considered unsafe.
  
- Maybe a way to load the coordinates of objects faster and a way to avoid coordinates being displayed on each other, where a type of offset may be added in implementation to avoid that.
  
- Maybe show the number of objects and keep updating the plot.
  
- Though this may be a minor feature, but maybe add a label for deleting objects. 
