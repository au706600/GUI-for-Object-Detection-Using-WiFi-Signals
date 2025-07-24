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

It enables users to simulate signal reflections, estimate object positions using MSE, and visualize the results through plots, histograms, and ROC curves. The **Target-Absent vs Target-Present Histogram** and

**ROC curve** are used to evaluate, whether or not, there is an object present at the estimated coordinate through evaluating the performance of the detector (See **Algorithms and Methods** section). By having 

the two statistical methods, it essentially helps us compare the performance of our detection algorithm.

## ⚙️ System Description

The system simulates a 2D indoor channel model with:

- A **transmitter (TX)** equipped with `M` antennas.
  
- A **target object** with unknown position on a 2D plane.
  
- A **Reconfigurable Intelligent Surface (RIS)** with `N` elements, designed to reflect signals using phase shifts.

The RIS act as the passive element that reflects the signal to the object of interest with
applied phase shifts, which enables the possibility of detecting changes.

In our project, we only consider a single transmitter and RIS at locations (3,0) and (0,3), respectively
for the sake of simplicity.


## 🧰 Prerequisites

- Python 3.8+ (Python 3.12.1 recommended)
  
- pip (Python package manager)
  
- Visual Studio Code (Recommended)


## ✨ Features

- **User Login System** – Simple session-based authentication with predefined credentials
  
- **Interactive Web GUI** – Built with HTML/CSS and Flask
  
- **Signal Reflection Simulation** – Models how WiFi signals interact with objects and RIS
  
- **Object Localization** – Estimates object position using Mean Squared Error (MSE)
  
- **Statistical Visualization** – ROC curves and histograms to evaluate detection performance


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
### After performing the Installation steps and go to http://127.0.0.1:5000/login, we come to the page shown below, where we subsequently input the login: 

<img width="1900" height="937" alt="image" src="https://github.com/user-attachments/assets/7552bb3d-8f44-48ee-8632-ee6a04bf8955" />

### After being authenticated, we navigate to the home page: 

<img width="1901" height="943" alt="image" src="https://github.com/user-attachments/assets/d82625cd-4921-42ef-81f6-fa18714c73f4" />
<img width="1920" height="941" alt="image" src="https://github.com/user-attachments/assets/79842e18-651b-49d8-a883-b032a00e8ca6" />
<img width="1896" height="756" alt="image" src="https://github.com/user-attachments/assets/85bfc48c-295b-49e9-a190-13a39621912f" />

### As an example, if the user inputs x = 4 and y = 7 in input x and input 7 as true coordinates, then the corresponding estimate of object position is shown on graph with the corresponding histogram and curve: 

<img width="1898" height="936" alt="image" src="https://github.com/user-attachments/assets/8725c425-ae11-4669-9948-8f136bb347eb" />
<img width="1897" height="631" alt="image" src="https://github.com/user-attachments/assets/b79c38e5-1ed8-48ff-bcf0-f0c2284484c1" />
<img width="1891" height="757" alt="image" src="https://github.com/user-attachments/assets/fdeca162-522a-4c1b-8a4b-03a2b15321a4" />


## 📌 Future Work

- Tracking objects when moving for true and estimated localization would be a nice-to-have feature.
  
- More secure login with storing credentials in database as shown in the **📦 Installation** section. As an extension, whenever fan/route is stored on "history" part on web browser,
  the user can basically cheat by changing the route to the desired route location, and the user can access the page.
  If history of the route is cleared from web browser, the user has to login with correct credentials, otherwise login is not possible.

- Designing a new algorithm, where we include RIS (ris set to 0) to enhance the signal.

- Modify existing code to maybe implement system so that it is more dynamic. 
  
- Register users safely in database, since in current project, we store it manually in python web framework Flask, which is considered unsafe.
  
- Maybe a way to load the coordinates of objects faster and a way to avoid coordinates being displayed on each other, where a type of offset may be added in implementation to avoid that.
  
- Maybe show the number of objects and keep updating the plot.
  
- Though this may be a minor feature, but maybe add a label for deleting objects. 
