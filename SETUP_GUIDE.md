# Setup and Run Instructions

This guide provides step-by-step instructions to clone the repository, install all required Python packages, and run the Ship Detection dashboard.

## 1. Clone the Repository

First, clone the project repository to your local machine and navigate into it:

```bash
git clone <repository_url>
cd ship-detection-sar
```
*(Replace `<repository_url>` with the actual URL of the repository).*

## 2. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage your dependencies.

**For Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**For macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Required Packages and Libraries

Run the following command to install all necessary Python packages and libraries specified in `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Run the Application

Once all the libraries are successfully installed, you can start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

After running the command, a browser window should open automatically showing the Ship Detection Dashboard.

---

### Additional Commands (Optional)

**To train the model on your dataset:**
```bash
python scripts/train.py --data data/yolo_format/dataset.yaml --epochs 50
```

**To evaluate the model:**
```bash
python scripts/evaluate.py --weights models/yolov8n_sar.pt --data data/yolo_format/dataset.yaml
```
