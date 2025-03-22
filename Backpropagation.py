{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "93aa4919-220b-4864-9c89-c39448a01c94",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Defaulting to user installation because normal site-packages is not writeable\n",
      "Requirement already satisfied: numpy in c:\\programdata\\anaconda3\\lib\\site-packages (1.26.4)\n",
      "Requirement already satisfied: pandas in c:\\programdata\\anaconda3\\lib\\site-packages (2.2.2)\n",
      "Collecting tensorflow\n",
      "  Using cached tensorflow-2.19.0-cp312-cp312-win_amd64.whl.metadata (4.1 kB)\n",
      "Requirement already satisfied: scikit-learn in c:\\programdata\\anaconda3\\lib\\site-packages (1.5.1)\n",
      "Requirement already satisfied: matplotlib in c:\\programdata\\anaconda3\\lib\\site-packages (3.9.2)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\programdata\\anaconda3\\lib\\site-packages (from pandas) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from pandas) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\programdata\\anaconda3\\lib\\site-packages (from pandas) (2023.3)\n",
      "Collecting absl-py>=1.0.0 (from tensorflow)\n",
      "  Using cached absl_py-2.2.0-py3-none-any.whl.metadata (2.4 kB)\n",
      "Collecting astunparse>=1.6.0 (from tensorflow)\n",
      "  Using cached astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)\n",
      "Collecting flatbuffers>=24.3.25 (from tensorflow)\n",
      "  Using cached flatbuffers-25.2.10-py2.py3-none-any.whl.metadata (875 bytes)\n",
      "Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow)\n",
      "  Using cached gast-0.6.0-py3-none-any.whl.metadata (1.3 kB)\n",
      "Collecting google-pasta>=0.1.1 (from tensorflow)\n",
      "  Using cached google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)\n",
      "Collecting libclang>=13.0.0 (from tensorflow)\n",
      "  Using cached libclang-18.1.1-py2.py3-none-win_amd64.whl.metadata (5.3 kB)\n",
      "Collecting opt-einsum>=2.3.2 (from tensorflow)\n",
      "  Using cached opt_einsum-3.4.0-py3-none-any.whl.metadata (6.3 kB)\n",
      "Requirement already satisfied: packaging in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow) (24.1)\n",
      "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow) (4.25.3)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow) (2.32.3)\n",
      "Requirement already satisfied: setuptools in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow) (75.1.0)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow) (1.16.0)\n",
      "Collecting termcolor>=1.1.0 (from tensorflow)\n",
      "  Using cached termcolor-2.5.0-py3-none-any.whl.metadata (6.1 kB)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow) (4.11.0)\n",
      "Requirement already satisfied: wrapt>=1.11.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow) (1.14.1)\n",
      "Collecting grpcio<2.0,>=1.24.3 (from tensorflow)\n",
      "  Using cached grpcio-1.71.0-cp312-cp312-win_amd64.whl.metadata (4.0 kB)\n",
      "Collecting tensorboard~=2.19.0 (from tensorflow)\n",
      "  Using cached tensorboard-2.19.0-py3-none-any.whl.metadata (1.8 kB)\n",
      "Collecting keras>=3.5.0 (from tensorflow)\n",
      "  Using cached keras-3.9.0-py3-none-any.whl.metadata (6.1 kB)\n",
      "Requirement already satisfied: h5py>=3.11.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow) (3.11.0)\n",
      "Collecting ml-dtypes<1.0.0,>=0.5.1 (from tensorflow)\n",
      "  Using cached ml_dtypes-0.5.1-cp312-cp312-win_amd64.whl.metadata (22 kB)\n",
      "Requirement already satisfied: scipy>=1.6.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from scikit-learn) (1.13.1)\n",
      "Requirement already satisfied: joblib>=1.2.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from scikit-learn) (1.4.2)\n",
      "Requirement already satisfied: threadpoolctl>=3.1.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from scikit-learn) (3.5.0)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (1.2.0)\n",
      "Requirement already satisfied: cycler>=0.10 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (4.51.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (1.4.4)\n",
      "Requirement already satisfied: pillow>=8 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (10.4.0)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (3.1.2)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from astunparse>=1.6.0->tensorflow) (0.44.0)\n",
      "Requirement already satisfied: rich in c:\\programdata\\anaconda3\\lib\\site-packages (from keras>=3.5.0->tensorflow) (13.7.1)\n",
      "Collecting namex (from keras>=3.5.0->tensorflow)\n",
      "  Using cached namex-0.0.8-py3-none-any.whl.metadata (246 bytes)\n",
      "Collecting optree (from keras>=3.5.0->tensorflow)\n",
      "  Using cached optree-0.14.1-cp312-cp312-win_amd64.whl.metadata (50 kB)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow) (2.2.3)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow) (2024.8.30)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorboard~=2.19.0->tensorflow) (3.4.1)\n",
      "Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard~=2.19.0->tensorflow)\n",
      "  Using cached tensorboard_data_server-0.7.2-py3-none-any.whl.metadata (1.1 kB)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorboard~=2.19.0->tensorflow) (3.0.3)\n",
      "Requirement already satisfied: MarkupSafe>=2.1.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from werkzeug>=1.0.1->tensorboard~=2.19.0->tensorflow) (2.1.3)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from rich->keras>=3.5.0->tensorflow) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from rich->keras>=3.5.0->tensorflow) (2.15.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.5.0->tensorflow) (0.1.0)\n",
      "Using cached tensorflow-2.19.0-cp312-cp312-win_amd64.whl (376.0 MB)\n",
      "Using cached absl_py-2.2.0-py3-none-any.whl (276 kB)\n",
      "Using cached astunparse-1.6.3-py2.py3-none-any.whl (12 kB)\n",
      "Using cached flatbuffers-25.2.10-py2.py3-none-any.whl (30 kB)\n",
      "Using cached gast-0.6.0-py3-none-any.whl (21 kB)\n",
      "Using cached google_pasta-0.2.0-py3-none-any.whl (57 kB)\n",
      "Using cached grpcio-1.71.0-cp312-cp312-win_amd64.whl (4.3 MB)\n",
      "Using cached keras-3.9.0-py3-none-any.whl (1.3 MB)\n",
      "Using cached libclang-18.1.1-py2.py3-none-win_amd64.whl (26.4 MB)\n",
      "Using cached ml_dtypes-0.5.1-cp312-cp312-win_amd64.whl (210 kB)\n",
      "Using cached opt_einsum-3.4.0-py3-none-any.whl (71 kB)\n",
      "Downloading tensorboard-2.19.0-py3-none-any.whl (5.5 MB)\n",
      "   ---------------------------------------- 0.0/5.5 MB ? eta -:--:--\n",
      "   - -------------------------------------- 0.3/5.5 MB ? eta -:--:--\n",
      "   --- ------------------------------------ 0.5/5.5 MB 1.9 MB/s eta 0:00:03\n",
      "   ------- -------------------------------- 1.0/5.5 MB 2.0 MB/s eta 0:00:03\n",
      "   ----------- ---------------------------- 1.6/5.5 MB 2.0 MB/s eta 0:00:03\n",
      "   ------------- -------------------------- 1.8/5.5 MB 1.9 MB/s eta 0:00:02\n",
      "   ----------------- ---------------------- 2.4/5.5 MB 1.9 MB/s eta 0:00:02\n",
      "   ------------------- -------------------- 2.6/5.5 MB 1.9 MB/s eta 0:00:02\n",
      "   ---------------------- ----------------- 3.1/5.5 MB 1.9 MB/s eta 0:00:02\n",
      "   ------------------------ --------------- 3.4/5.5 MB 1.9 MB/s eta 0:00:02\n",
      "   ---------------------------- ----------- 3.9/5.5 MB 1.9 MB/s eta 0:00:01\n",
      "   ------------------------------ --------- 4.2/5.5 MB 1.9 MB/s eta 0:00:01\n",
      "   ---------------------------------- ----- 4.7/5.5 MB 1.9 MB/s eta 0:00:01\n",
      "   -------------------------------------- - 5.2/5.5 MB 1.9 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 5.5/5.5 MB 1.9 MB/s eta 0:00:00\n",
      "Downloading termcolor-2.5.0-py3-none-any.whl (7.8 kB)\n",
      "Downloading tensorboard_data_server-0.7.2-py3-none-any.whl (2.4 kB)\n",
      "Downloading namex-0.0.8-py3-none-any.whl (5.8 kB)\n",
      "Downloading optree-0.14.1-cp312-cp312-win_amd64.whl (306 kB)\n",
      "Installing collected packages: namex, libclang, flatbuffers, termcolor, tensorboard-data-server, optree, opt-einsum, ml-dtypes, grpcio, google-pasta, gast, astunparse, absl-py, tensorboard, keras, tensorflow\n",
      "Successfully installed absl-py-2.2.0 astunparse-1.6.3 flatbuffers-25.2.10 gast-0.6.0 google-pasta-0.2.0 grpcio-1.71.0 keras-3.9.0 libclang-18.1.1 ml-dtypes-0.5.1 namex-0.0.8 opt-einsum-3.4.0 optree-0.14.1 tensorboard-2.19.0 tensorboard-data-server-0.7.2 tensorflow-2.19.0 termcolor-2.5.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  WARNING: The script tensorboard.exe is installed in 'C:\\Users\\Ismri\\AppData\\Roaming\\Python\\Python312\\Scripts' which is not on PATH.\n",
      "  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.\n",
      "  WARNING: The scripts import_pb_to_tensorboard.exe, saved_model_cli.exe, tensorboard.exe, tf_upgrade_v2.exe, tflite_convert.exe and toco.exe are installed in 'C:\\Users\\Ismri\\AppData\\Roaming\\Python\\Python312\\Scripts' which is not on PATH.\n",
      "  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.\n"
     ]
    }
   ],
   "source": [
    "pip install numpy pandas tensorflow scikit-learn matplotlib\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "bad9783d-2d8b-4c54-9c5c-8154d6e41e8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "c7093cc7-650d-4291-9ec4-e05c3c4c3c1b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Ismri\\AppData\\Local\\Temp\\ipykernel_27932\\651392061.py:11: FutureWarning: The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version. Use ``sep='\\s+'`` instead\n",
      "  data = pd.read_csv(url, delim_whitespace=True, header=None, names=column_names)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Area  Perimeter  Compactness  Kernel Length  Kernel Width  \\\n",
      "0  15.26      14.84       0.8710          5.763         3.312   \n",
      "1  14.88      14.57       0.8811          5.554         3.333   \n",
      "2  14.29      14.09       0.9050          5.291         3.337   \n",
      "3  13.84      13.94       0.8955          5.324         3.379   \n",
      "4  16.14      14.99       0.9034          5.658         3.562   \n",
      "\n",
      "   Asymmetry Coefficient  Groove Length  Class  \n",
      "0                  2.221          5.220      1  \n",
      "1                  1.018          4.956      1  \n",
      "2                  2.699          4.825      1  \n",
      "3                  2.259          4.805      1  \n",
      "4                  1.355          5.175      1  \n"
     ]
    }
   ],
   "source": [
    "# Load dataset from UCI Repository\n",
    "url = \"https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt\"\n",
    "\n",
    "# Define column names\n",
    "column_names = [\n",
    "    \"Area\", \"Perimeter\", \"Compactness\", \"Kernel Length\", \"Kernel Width\",\n",
    "    \"Asymmetry Coefficient\", \"Groove Length\", \"Class\"\n",
    "]\n",
    "\n",
    "# Load dataset\n",
    "data = pd.read_csv(url, delim_whitespace=True, header=None, names=column_names)\n",
    "\n",
    "# Display first few rows\n",
    "print(data.head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d290c13d-7bea-4c5c-8067-6e888110274a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Separate features (X) and labels (y)\n",
    "X = data.iloc[:, :-1].values  # All columns except the last one (features)\n",
    "y = data.iloc[:, -1].values   # Last column (labels)\n",
    "\n",
    "# Encode labels (1,2,3 → 0,1,2)\n",
    "label_encoder = LabelEncoder()\n",
    "y = label_encoder.fit_transform(y)\n",
    "\n",
    "# Standardize the features\n",
    "scaler = StandardScaler()\n",
    "X = scaler.fit_transform(X)\n",
    "\n",
    "# Split into train (80%) and test (20%)\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "90b3f222-e030-4ad0-b6bc-a4cfad515f99",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Ismri\\AppData\\Roaming\\Python\\Python312\\site-packages\\keras\\src\\layers\\core\\dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>)             │           <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">8</span>)              │           <span style=\"color: #00af00; text-decoration-color: #00af00\">136</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3</span>)              │            <span style=\"color: #00af00; text-decoration-color: #00af00\">27</span> │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m16\u001b[0m)             │           \u001b[38;5;34m128\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m8\u001b[0m)              │           \u001b[38;5;34m136\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_2 (\u001b[38;5;33mDense\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m3\u001b[0m)              │            \u001b[38;5;34m27\u001b[0m │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">291</span> (1.14 KB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m291\u001b[0m (1.14 KB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">291</span> (1.14 KB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m291\u001b[0m (1.14 KB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Define the neural network model\n",
    "model = Sequential([\n",
    "    Dense(16, activation='relu', input_shape=(7,)),  # First hidden layer\n",
    "    Dense(8, activation='relu'),  # Second hidden layer\n",
    "    Dense(3, activation='softmax')  # Output layer (3 classes)\n",
    "])\n",
    "\n",
    "# Compile the model (Backpropagation implemented internally)\n",
    "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# Print model summary\n",
    "model.summary()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f65d4162-4f84-4fb5-a53e-d7d32c0c04a5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 10ms/step - accuracy: 0.3576 - loss: 1.0635 - val_accuracy: 0.5000 - val_loss: 0.9820\n",
      "Epoch 2/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.6288 - loss: 0.9668 - val_accuracy: 0.7381 - val_loss: 0.8928\n",
      "Epoch 3/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.7356 - loss: 0.8643 - val_accuracy: 0.7619 - val_loss: 0.8058\n",
      "Epoch 4/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.8602 - loss: 0.7305 - val_accuracy: 0.8333 - val_loss: 0.7099\n",
      "Epoch 5/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.8346 - loss: 0.7219 - val_accuracy: 0.8571 - val_loss: 0.6098\n",
      "Epoch 6/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.8855 - loss: 0.5829 - val_accuracy: 0.8810 - val_loss: 0.5198\n",
      "Epoch 7/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9101 - loss: 0.4993 - val_accuracy: 0.8810 - val_loss: 0.4525\n",
      "Epoch 8/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.8876 - loss: 0.4367 - val_accuracy: 0.8810 - val_loss: 0.4054\n",
      "Epoch 9/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9205 - loss: 0.3293 - val_accuracy: 0.8810 - val_loss: 0.3691\n",
      "Epoch 10/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9377 - loss: 0.2846 - val_accuracy: 0.8810 - val_loss: 0.3437\n",
      "Epoch 11/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9286 - loss: 0.2904 - val_accuracy: 0.9048 - val_loss: 0.3295\n",
      "Epoch 12/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9114 - loss: 0.2804 - val_accuracy: 0.9286 - val_loss: 0.3167\n",
      "Epoch 13/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9284 - loss: 0.2492 - val_accuracy: 0.9286 - val_loss: 0.3032\n",
      "Epoch 14/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9137 - loss: 0.2564 - val_accuracy: 0.9286 - val_loss: 0.2964\n",
      "Epoch 15/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9020 - loss: 0.2313 - val_accuracy: 0.9286 - val_loss: 0.2870\n",
      "Epoch 16/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9306 - loss: 0.1915 - val_accuracy: 0.9286 - val_loss: 0.2891\n",
      "Epoch 17/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9077 - loss: 0.2095 - val_accuracy: 0.9286 - val_loss: 0.2802\n",
      "Epoch 18/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9274 - loss: 0.1788 - val_accuracy: 0.9286 - val_loss: 0.2725\n",
      "Epoch 19/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9628 - loss: 0.1615 - val_accuracy: 0.9048 - val_loss: 0.2633\n",
      "Epoch 20/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9394 - loss: 0.1868 - val_accuracy: 0.9286 - val_loss: 0.2639\n",
      "Epoch 21/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9228 - loss: 0.2258 - val_accuracy: 0.9286 - val_loss: 0.2580\n",
      "Epoch 22/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9483 - loss: 0.1800 - val_accuracy: 0.9048 - val_loss: 0.2547\n",
      "Epoch 23/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9622 - loss: 0.1598 - val_accuracy: 0.9048 - val_loss: 0.2508\n",
      "Epoch 24/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9144 - loss: 0.2080 - val_accuracy: 0.9286 - val_loss: 0.2537\n",
      "Epoch 25/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9544 - loss: 0.1647 - val_accuracy: 0.9048 - val_loss: 0.2450\n",
      "Epoch 26/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9491 - loss: 0.1937 - val_accuracy: 0.9048 - val_loss: 0.2417\n",
      "Epoch 27/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9355 - loss: 0.1806 - val_accuracy: 0.9048 - val_loss: 0.2392\n",
      "Epoch 28/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9645 - loss: 0.1268 - val_accuracy: 0.9286 - val_loss: 0.2419\n",
      "Epoch 29/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9456 - loss: 0.1902 - val_accuracy: 0.9048 - val_loss: 0.2309\n",
      "Epoch 30/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9281 - loss: 0.1847 - val_accuracy: 0.9048 - val_loss: 0.2310\n",
      "Epoch 31/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9531 - loss: 0.1337 - val_accuracy: 0.9048 - val_loss: 0.2276\n",
      "Epoch 32/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9327 - loss: 0.1752 - val_accuracy: 0.9048 - val_loss: 0.2304\n",
      "Epoch 33/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9547 - loss: 0.1292 - val_accuracy: 0.9048 - val_loss: 0.2239\n",
      "Epoch 34/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9497 - loss: 0.1201 - val_accuracy: 0.9048 - val_loss: 0.2254\n",
      "Epoch 35/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9536 - loss: 0.1562 - val_accuracy: 0.9048 - val_loss: 0.2156\n",
      "Epoch 36/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9519 - loss: 0.1382 - val_accuracy: 0.9048 - val_loss: 0.2203\n",
      "Epoch 37/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9659 - loss: 0.1018 - val_accuracy: 0.9048 - val_loss: 0.2191\n",
      "Epoch 38/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9734 - loss: 0.0965 - val_accuracy: 0.9048 - val_loss: 0.2152\n",
      "Epoch 39/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9550 - loss: 0.1359 - val_accuracy: 0.9048 - val_loss: 0.2130\n",
      "Epoch 40/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9689 - loss: 0.1213 - val_accuracy: 0.9048 - val_loss: 0.2133\n",
      "Epoch 41/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9459 - loss: 0.1397 - val_accuracy: 0.9048 - val_loss: 0.2098\n",
      "Epoch 42/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9519 - loss: 0.1199 - val_accuracy: 0.9048 - val_loss: 0.2103\n",
      "Epoch 43/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9744 - loss: 0.0911 - val_accuracy: 0.9048 - val_loss: 0.2095\n",
      "Epoch 44/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9624 - loss: 0.0895 - val_accuracy: 0.9048 - val_loss: 0.2086\n",
      "Epoch 45/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9143 - loss: 0.1865 - val_accuracy: 0.9048 - val_loss: 0.2064\n",
      "Epoch 46/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9546 - loss: 0.1295 - val_accuracy: 0.9048 - val_loss: 0.2045\n",
      "Epoch 47/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9581 - loss: 0.1243 - val_accuracy: 0.8810 - val_loss: 0.2075\n",
      "Epoch 48/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9741 - loss: 0.0937 - val_accuracy: 0.8810 - val_loss: 0.2057\n",
      "Epoch 49/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9127 - loss: 0.1669 - val_accuracy: 0.8810 - val_loss: 0.2053\n",
      "Epoch 50/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9543 - loss: 0.1049 - val_accuracy: 0.9048 - val_loss: 0.2021\n",
      "Epoch 51/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9462 - loss: 0.1091 - val_accuracy: 0.8810 - val_loss: 0.2014\n",
      "Epoch 52/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9594 - loss: 0.1116 - val_accuracy: 0.8810 - val_loss: 0.2016\n",
      "Epoch 53/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9508 - loss: 0.1141 - val_accuracy: 0.8810 - val_loss: 0.1990\n",
      "Epoch 54/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9534 - loss: 0.1177 - val_accuracy: 0.8810 - val_loss: 0.1996\n",
      "Epoch 55/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9438 - loss: 0.1159 - val_accuracy: 0.8810 - val_loss: 0.1961\n",
      "Epoch 56/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9670 - loss: 0.0862 - val_accuracy: 0.8810 - val_loss: 0.1987\n",
      "Epoch 57/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9396 - loss: 0.1187 - val_accuracy: 0.8810 - val_loss: 0.1959\n",
      "Epoch 58/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9630 - loss: 0.0996 - val_accuracy: 0.8810 - val_loss: 0.1963\n",
      "Epoch 59/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9715 - loss: 0.0830 - val_accuracy: 0.8810 - val_loss: 0.1950\n",
      "Epoch 60/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9486 - loss: 0.0955 - val_accuracy: 0.8810 - val_loss: 0.1941\n",
      "Epoch 61/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9771 - loss: 0.0853 - val_accuracy: 0.8810 - val_loss: 0.1910\n",
      "Epoch 62/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9774 - loss: 0.0676 - val_accuracy: 0.8810 - val_loss: 0.1927\n",
      "Epoch 63/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9462 - loss: 0.1049 - val_accuracy: 0.8810 - val_loss: 0.1883\n",
      "Epoch 64/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9756 - loss: 0.0722 - val_accuracy: 0.8810 - val_loss: 0.1910\n",
      "Epoch 65/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9767 - loss: 0.0782 - val_accuracy: 0.8810 - val_loss: 0.1855\n",
      "Epoch 66/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9849 - loss: 0.0739 - val_accuracy: 0.8810 - val_loss: 0.1865\n",
      "Epoch 67/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9783 - loss: 0.0593 - val_accuracy: 0.8810 - val_loss: 0.1891\n",
      "Epoch 68/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9709 - loss: 0.0989 - val_accuracy: 0.8810 - val_loss: 0.1857\n",
      "Epoch 69/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9664 - loss: 0.0829 - val_accuracy: 0.8810 - val_loss: 0.1832\n",
      "Epoch 70/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9584 - loss: 0.1129 - val_accuracy: 0.8810 - val_loss: 0.1814\n",
      "Epoch 71/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9856 - loss: 0.0593 - val_accuracy: 0.8810 - val_loss: 0.1808\n",
      "Epoch 72/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9838 - loss: 0.0633 - val_accuracy: 0.8810 - val_loss: 0.1820\n",
      "Epoch 73/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9671 - loss: 0.0875 - val_accuracy: 0.8810 - val_loss: 0.1816\n",
      "Epoch 74/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9842 - loss: 0.0550 - val_accuracy: 0.8810 - val_loss: 0.1813\n",
      "Epoch 75/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9839 - loss: 0.0685 - val_accuracy: 0.8810 - val_loss: 0.1777\n",
      "Epoch 76/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9835 - loss: 0.0699 - val_accuracy: 0.8810 - val_loss: 0.1770\n",
      "Epoch 77/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9875 - loss: 0.0578 - val_accuracy: 0.8810 - val_loss: 0.1765\n",
      "Epoch 78/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9668 - loss: 0.1003 - val_accuracy: 0.9048 - val_loss: 0.1809\n",
      "Epoch 79/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9868 - loss: 0.0871 - val_accuracy: 0.8810 - val_loss: 0.1752\n",
      "Epoch 80/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9792 - loss: 0.0432 - val_accuracy: 0.8810 - val_loss: 0.1717\n",
      "Epoch 81/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9836 - loss: 0.0584 - val_accuracy: 0.8810 - val_loss: 0.1704\n",
      "Epoch 82/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9844 - loss: 0.0619 - val_accuracy: 0.8810 - val_loss: 0.1743\n",
      "Epoch 83/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9747 - loss: 0.0596 - val_accuracy: 0.8810 - val_loss: 0.1688\n",
      "Epoch 84/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9854 - loss: 0.0537 - val_accuracy: 0.8810 - val_loss: 0.1734\n",
      "Epoch 85/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9705 - loss: 0.0843 - val_accuracy: 0.8810 - val_loss: 0.1688\n",
      "Epoch 86/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9844 - loss: 0.0492 - val_accuracy: 0.8810 - val_loss: 0.1679\n",
      "Epoch 87/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9608 - loss: 0.0744 - val_accuracy: 0.8810 - val_loss: 0.1670\n",
      "Epoch 88/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9905 - loss: 0.0496 - val_accuracy: 0.8810 - val_loss: 0.1664\n",
      "Epoch 89/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9720 - loss: 0.0545 - val_accuracy: 0.8810 - val_loss: 0.1681\n",
      "Epoch 90/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9911 - loss: 0.0408 - val_accuracy: 0.8810 - val_loss: 0.1646\n",
      "Epoch 91/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9744 - loss: 0.0807 - val_accuracy: 0.8810 - val_loss: 0.1643\n",
      "Epoch 92/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9849 - loss: 0.0600 - val_accuracy: 0.8810 - val_loss: 0.1647\n",
      "Epoch 93/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9730 - loss: 0.0770 - val_accuracy: 0.8810 - val_loss: 0.1617\n",
      "Epoch 94/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9902 - loss: 0.0620 - val_accuracy: 0.8810 - val_loss: 0.1609\n",
      "Epoch 95/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9879 - loss: 0.0433 - val_accuracy: 0.8810 - val_loss: 0.1626\n",
      "Epoch 96/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9973 - loss: 0.0489 - val_accuracy: 0.8810 - val_loss: 0.1635\n",
      "Epoch 97/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9911 - loss: 0.0475 - val_accuracy: 0.8810 - val_loss: 0.1622\n",
      "Epoch 98/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9863 - loss: 0.0467 - val_accuracy: 0.8810 - val_loss: 0.1574\n",
      "Epoch 99/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9757 - loss: 0.0811 - val_accuracy: 0.8810 - val_loss: 0.1592\n",
      "Epoch 100/100\n",
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - accuracy: 0.9769 - loss: 0.0797 - val_accuracy: 0.8810 - val_loss: 0.1576\n"
     ]
    }
   ],
   "source": [
    "# Train the model\n",
    "history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1a68a24c-280b-474e-a3af-0d0830c43de6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 42ms/step\n",
      "Test Accuracy: 0.88\n"
     ]
    }
   ],
   "source": [
    "# Predict on test data\n",
    "y_pred = np.argmax(model.predict(X_test), axis=1)\n",
    "\n",
    "# Calculate accuracy\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"Test Accuracy: {accuracy:.2f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "002824bf-b2a6-442a-b36d-e4f6be9f9fad",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHFCAYAAAAOmtghAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABjuUlEQVR4nO3deXhU1f3H8fcsmez7npCEACI7KIgsLigWBEtFXKhVEZcqVVS0tmrdkNZiF5Vaf2CxCnWntGptxQVUEEEQQTZZhZAEkhCy79vM/f1xh5EIhCQkmSyf1/PcJ8mdO3dObmnz6Tnfc47FMAwDERERkU7C6u0GiIiIiLQkhRsRERHpVBRuREREpFNRuBEREZFOReFGREREOhWFGxEREelUFG5ERESkU1G4ERERkU5F4UZEREQ6FYUbEWnQ4sWLsVgsfP31195uSqOsXr2aa665hsTERBwOB6GhoYwaNYoFCxZQXl7u7eaJSBtQuBGRTuPxxx/nggsu4NChQ/z2t79l+fLlvPXWW4wdO5bZs2fzyCOPeLuJItIG7N5ugIhIS1i6dClz5szhlltu4cUXX8RisXhemzBhAr/+9a/58ssvW+SzKioqCAgIaJF7iUjLU8+NiLSIL774grFjxxIcHExAQACjRo3i/fffr3dNRUUF999/P6mpqfj5+REREcGwYcN48803Pdfs37+fn/70pyQkJODr60tsbCxjx45l8+bNDX7+nDlzCA8P57nnnqsXbI4KDg5m3LhxABw4cACLxcLixYuPu85isTB79mzPz7Nnz8ZisbBp0yauuuoqwsPD6dmzJ/PmzcNisfDdd98dd48HHngAh8NBXl6e59yKFSsYO3YsISEhBAQEMHr0aD755JMGfycRaR6FGxE5batWreLiiy+muLiYl156iTfffJPg4GAmTZrEkiVLPNfdd999LFiwgLvvvpsPP/yQV199lauvvpr8/HzPNRMnTmTjxo388Y9/ZPny5SxYsICzzjqLoqKik35+dnY227dvZ9y4ca3WozJlyhR69erF0qVLeeGFF7j++utxOBzHBSSn08lrr73GpEmTiIqKAuC1115j3LhxhISE8I9//IN//vOfREREMH78eAUckdZgiIg0YNGiRQZgbNiw4aTXjBgxwoiJiTFKS0s95+rq6owBAwYY3bp1M1wul2EYhjFgwABj8uTJJ71PXl6eARjz5s1rUhvXrVtnAMaDDz7YqOvT0tIMwFi0aNFxrwHG448/7vn58ccfNwDjscceO+7aKVOmGN26dTOcTqfn3LJlywzA+O9//2sYhmGUl5cbERERxqRJk+q91+l0GoMHDzaGDx/eqDaLSOOp50ZETkt5eTnr16/nqquuIigoyHPeZrNxww03cPDgQXbv3g3A8OHD+eCDD3jwwQdZuXIllZWV9e4VERFBz549+dOf/sQzzzzDN998g8vlatPf52SuvPLK487ddNNNHDx4kBUrVnjOLVq0iLi4OCZMmADA2rVrKSgo4MYbb6Surs5zuFwuLr30UjZs2KBZXCItTOFGRE5LYWEhhmEQHx9/3GsJCQkAnmGn5557jgceeIB3332Xiy66iIiICCZPnszevXsBs97lk08+Yfz48fzxj3/k7LPPJjo6mrvvvpvS0tKTtiE5ORmAtLS0lv71PE70+02YMIH4+HgWLVoEmM/ivffeY9q0adhsNgAOHz4MwFVXXYWPj0+94w9/+AOGYVBQUNBq7RbpijRbSkROS3h4OFarlezs7ONey8rKAvDUngQGBvLEE0/wxBNPcPjwYU8vzqRJk9i1axcAKSkpvPTSSwDs2bOHf/7zn8yePZuamhpeeOGFE7YhPj6egQMH8vHHHzdqJpOfnx8A1dXV9c4fW/vzQycqUj7aO/Xcc89RVFTEG2+8QXV1NTfddJPnmqO/+1//+ldGjBhxwnvHxsY22F4RaRr13IjIaQkMDOTcc8/l7bffrjfM5HK5eO211+jWrRu9e/c+7n2xsbFMnz6da6+9lt27d1NRUXHcNb179+aRRx5h4MCBbNq0qcF2PProoxQWFnL33XdjGMZxr5eVlfHxxx97PtvPz4+tW7fWu+Y///lPo37nY910001UVVXx5ptvsnjxYkaOHEmfPn08r48ePZqwsDB27NjBsGHDTng4HI4mf66InJx6bkSkUT799FMOHDhw3PmJEycyd+5cfvSjH3HRRRdx//3343A4mD9/Ptu3b+fNN9/09Hqce+65/PjHP2bQoEGEh4ezc+dOXn31VUaOHElAQABbt25l5syZXH311Zxxxhk4HA4+/fRTtm7dyoMPPthg+66++moeffRRfvvb37Jr1y5uueUWevbsSUVFBevXr+dvf/sbU6dOZdy4cVgsFq6//npefvllevbsyeDB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78+GDRs455xzyMjI4Fe/+hV9+ph/8M444wzP+zMyMrjyyisZOHAgAD169GhyG5pKw1ItyT0sRWE6vSIdgBbyExHxtj59+jBq1ChefvllAPbt28fq1au5+eabAXA6nTz55JMMGjSIyMhIgoKC+Pjjj+uts9aQnTt3kpyc7Ak2YK6o/0P/+te/OO+884iLiyMoKIhHH3200Z9x7GcNHjzYE2wARo8ejcvlYvfu3Z5z/fv3x2azeX6Oj48nNze3SZ917GcmJSXV23i6X79+hIWFsXPnTsBcp+7WW2/lkksu4amnnmLfvn2ea++++25+97vfMXr0aB5//HG2bt3arHY0hXpuWlJwvNlVWlvBgIBCQNPBRaST8wkwe1G88blNcMsttzBz5kz+7//+j0WLFpGSksLYsWMBePrpp3n22WeZN28eAwcOJDAwkFmzZlFT07gtdE60XtsPN3Net24dP/3pT3niiScYP348oaGhvPXWWzz99NNN+j0MwzjpRtHHnv/hlkQWiwWXq3kTXE72mceenz17Nj/72c94//33+eCDD3j88cd56623uOKKK7j11lsZP34877//Ph9//DFz587l6aef5q677mpWexpDPTctyWLx9N70spmFVpoOLiKdmsViDg+19dGIeptjXXPNNdhsNt544w3+8Y9/cNNNN3n+MK9evZrLL7+c66+/nsGDB9OjRw/27t3b6Hv369ePjIwMsrK+D3lffvllvWvWrFlDSkoKDz/8MMOGDeOMM844bgaXw+HA6XSe8rM2b95Mefn3f1vWrFmD1Wqld+/ejW5zUxz9/TIzMz3nduzYQXFxMX379vWc6927N/feey8ff/wxU6ZMYdGiRZ7XkpKSmDFjBm+//Ta//OUvefHFF1ulrUcp3LQ0d91NvPMQAEdKqymtan6FuoiInL6goCCmTp3Kb37zG7Kyspg+fbrntV69erF8+XLWrl3Lzp07uf3228nJyWn0vS+55BLOPPNMpk2bxpYtW1i9ejUPP/xwvWt69epFRkYGb731Fvv27eO5557jnXfeqXdN9+7dPXss5uXlUV1dfdxnXXfddfj5+XHjjTeyfft2PvvsM+666y5uuOEGT71NczmdTs9+j0ePHTt2cMkllzBo0CCuu+46Nm3axFdffcW0adO48MILGTZsGJWVlcycOZOVK1eSnp7OmjVr2LBhgyf4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8MMPT6s9GpY6TTU1NWzcuJFx48bVOz9u3DjWrl3rpVZ1TsXFxQBEREQAkJaWRk5OTr1n7+vry4UXXqhn30x33nknl112GZdcckm983rWLee9995j2LBhXH311cTExHDWWWfx4osvel7Xs2455513Hp988gl79uwBYMuWLXzxxRdMnDgR0LNuTY15ths3bqS2trbeNQkJCQwYMOC0n3+X2zizpeXl5eF0OomNja13PjY2lpycHC+1qvMxDIP77ruP8847jwEDBgB4nu+Jnn16enqbt7Gje+utt9i0aRMbNmw47jU965azf/9+FixYwH333cdvfvMbvvrqK+6++258fX2ZNm2annULeuCBByguLqZPnz7YbDacTidPPvkk1157LaB/162pMc82JycHh8NBeHj4cdec7t9PhZsWYrFY6v1sGMZx56T5Zs6cydatW/niiy+Oe03P/vRlZmZyzz338PHHH+Pn53fS6/SsT5/L5WLYsGH8/ve/B+Css87i22+/ZcGCBUybNs1znZ716VuyZAmvvfYab7zxBv3792fz5s3MmjWLhIQEbrzxRs91etatpznPtiWev4alTlNUVBQ2m+24lJmbm3tcYpXmueuuu3jvvff47LPP6Natm+d8XFwcgJ59C9i4cSO5ubkMHToUu92O3W5n1apVPPfcc9jtds/z1LM+ffHx8fTr16/eub59+5KRkQHo33VL+tWvfsWDDz7IT3/6UwYOHMgNN9zAvffey9y5cwE969bUmGcbFxdHTU0NhYWFJ72muRRuTpPD4WDo0KEsX7683vnly5czatQoL7WqczAMg5kzZ/L222/z6aefkpqaWu/11NRU4uLi6j37mpoaVq1apWffRGPHjmXbtm1s3rzZcwwbNozrrruOzZs306NHDz3rFjJ69OjjljTYs2cPKSkpgP5dt6SKigqs1vp/5mw2m2cquJ5162nMsx06dCg+Pj71rsnOzmb79u2n//xPqxxZDMP4fir4Sy+9ZOzYscOYNWuWERgYaBw4cMDbTevQfvGLXxihoaHGypUrjezsbM9RUVHhueapp54yQkNDjbffftvYtm2bce2112oaZws5draUYehZt5SvvvrKsNvtxpNPPmns3bvXeP31142AgADjtdde81yjZ90ybrzxRiMxMdEzFfztt982oqKijF//+teea/Ssm6+0tNT45ptvjG+++cYAjGeeecb45ptvPMugNObZzpgxw+jWrZuxYsUKY9OmTcbFF1+sqeDtyf/93/8ZKSkphsPhMM4++2zPdGVpPuCEx6JFizzXuFwu4/HHHzfi4uIMX19f44ILLjC2bdvmvUZ3Ij8MN3rWLee///2vMWDAAMPX19fo06ePsXDhwnqv61m3jJKSEuOee+4xkpOTDT8/P6NHjx7Gww8/bFRXV3uu0bNuvs8+++yE/xt94403GobRuGdbWVlpzJw504iIiDD8/f2NH//4x0ZGRsZpt81iGIZxen0/IiIiIu2Ham5ERESkU1G4ERERkU5F4UZEREQ6FYUbERER6VQUbkRERKRTUbgRERGRTkXhRkRERDoVhRsR6ZIsFgvvvvuut5shIq1A4UZE2tz06dOxWCzHHZdeeqm3myYinYDd2w0Qka7p0ksvZdGiRfXO+fr6eqk1ItKZqOdGRLzC19eXuLi4ekd4eDhgDhktWLCACRMm4O/vT2pqKkuXLq33/m3btnHxxRfj7+9PZGQkt912G2VlZfWuefnll+nfvz++vr7Ex8czc+bMeq/n5eVxxRVXEBAQwBlnnMF7773nea2wsJDrrruO6Oho/P39OeOMM44LYyLSPinciEi79Oijj3LllVeyZcsWrr/+eq699lp27twJQEVFBZdeeinh4eFs2LCBpUuXsmLFinrhZcGCBdx5553cdtttbNu2jffee49evXrV+4wnnniCa665hq1btzJx4kSuu+46CgoKPJ+/Y8cOPvjgA3bu3MmCBQuIiopquwcgIs132ltviog00Y033mjYbDYjMDCw3jFnzhzDMMwd4WfMmFHvPeeee67xi1/8wjAMw1i4cKERHh5ulJWVeV5///33DavVauTk5BiGYRgJCQnGww8/fNI2AMYjjzzi+bmsrMywWCzGBx98YBiGYUyaNMm46aabWuYXFpE2pZobEfGKiy66iAULFtQ7FxER4fl+5MiR9V4bOXIkmzdvBmDnzp0MHjyYwMBAz+ujR4/G5XKxe/duLBYLWVlZjB07tsE2DBo0yPN9YGAgwcHB5ObmAvCLX/yCK6+8kk2bNjFu3DgmT57MqFGjmvW7ikjbUrgREa8IDAw8bpjoVCwWCwCGYXi+P9E1/v7+jbqfj4/Pce91uVwATJgwgfT0dN5//31WrFjB2LFjufPOO/nzn//cpDaLSNtTzY2ItEvr1q077uc+ffoA0K9fPzZv3kx5ebnn9TVr1mC1WunduzfBwcF0796dTz755LTaEB0dzfTp03nttdeYN28eCxcuPK37iUjbUM+NiHhFdXU1OTk59c7Z7XZP0e7SpUsZNmwY5513Hq+//jpfffUVL730EgDXXXcdjz/+ODfeeCOzZ8/myJEj3HXXXdxwww3ExsYCMHv2bGbMmEFMTAwTJkygtLSUNWvWcNdddzWqfY899hhDhw6lf//+VFdX87///Y++ffu24BMQkdaicCMiXvHhhx8SHx9f79yZZ57Jrl27AHMm01tvvcUdd9xBXFwcr7/+Ov369QMgICCAjz76iHvuuYdzzjmHgIAArrzySp555hnPvW688Uaqqqp49tlnuf/++4mKiuKqq65qdPscDgcPPfQQBw4cwN/fn/PPP5+33nqrBX5zEWltFsMwDG83QkTkWBaLhXfeeYfJkyd7uyki0gGp5kZEREQ6FYUbERER6VRUcyMi7Y5Gy0XkdKjnRkRERDoVhRsRERHpVBRuREREpFNRuBEREZFOReFGREREOhWFGxEREelUFG5ERESkU1G4ERERkU5F4UZEREQ6lf8Hl5gLBEYntJMAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHFCAYAAAAOmtghAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABrb0lEQVR4nO3dd3RU5dYG8GcyyUx6SC+QSg2EmiDSpEpvClKkFxWRJnpBRCxcFD5R4CKCggRQQIoUEWmhVwUCgUBCL4GQEBIglbSZ9/tjkiFDCimTOcnw/NaaRXLmlH3O5Xo2+20yIYQAERERkZEwkToAIiIiIn1ickNERERGhckNERERGRUmN0RERGRUmNwQERGRUWFyQ0REREaFyQ0REREZFSY3REREZFSY3BAREZFRYXJDZMQWLVoEmUyGgIAAqUOplB48eIBPPvkE9evXh7W1NczNzVGzZk1MmjQJ165dkzo8IiqEjMsvEBmvRo0a4fz58wCAf/75B82aNZM4osrj1KlT6NGjB4QQGD9+PJo3bw6FQoErV65gzZo1uHjxIh4/fix1mERUACY3REbqzJkzaNq0Kbp3746///4b77zzDpYtWyZ1WAVKS0uDpaWl1GFoJSUloXbt2jAzM8OJEydQrVq1fPv88ccf6NevX5mvpVKpkJ2dDaVSWeZzEZEGm6WIjNSKFSsAAHPnzkWLFi2wfv16pKWl5dsvOjoa7777Ljw9PaFQKODh4YF+/frhwYMH2n2ePHmCjz76CH5+flAqlXBxcUG3bt1w+fJlAMChQ4cgk8lw6NAhnXPfvn0bMpkMq1at0m4bMWIErK2tER4ejk6dOsHGxgYdOnQAAISEhKB3796oVq0azM3NUaNGDbz33nuIj4/PF/fly5cxaNAguLq6QqlUwsvLC8OGDUNGRgZu374NU1NTzJkzJ99xR44cgUwmw6ZNmwp9dsuXL0dsbCy+/fbbAhMbADqJTdu2bdG2bdt8+4wYMQI+Pj75nse3336L2bNnw9fXF0qlEhs3boRCocDMmTMLvE+ZTIZFixZpt8XGxuK9995DtWrVoFAo4Ovri6+++grZ2dmF3hPRy8RU6gCISP+ePn2K33//HU2bNkVAQABGjRqFMWPGYNOmTRg+fLh2v+joaDRt2hRZWVn49NNP0aBBAyQkJGDPnj14/PgxXF1dkZycjFatWuH27duYNm0amjVrhpSUFBw5cgQxMTGoU6dOiePLzMxEr1698N577+GTTz7RvpRv3LiB5s2bY8yYMbCzs8Pt27cxf/58tGrVCuHh4TAzMwMAnD9/Hq1atYKTkxNmzZqFmjVrIiYmBtu3b0dmZiZ8fHzQq1cv/PTTT5g6dSrkcrn22osXL4aHhwfeeOONQuPbu3cv5HI5evbsWeJ7K45FixahVq1a+O6772Bra4uaNWuiR48eWL16Nb766iuYmDz7d+fKlSuhUCgwePBgAJrE5pVXXoGJiQk+//xzVK9eHSdPnsTs2bNx+/ZtrFy5slxiJqpUBBEZnV9//VUAED/99JMQQojk5GRhbW0tWrdurbPfqFGjhJmZmYiIiCj0XLNmzRIAREhISKH7HDx4UAAQBw8e1Nl+69YtAUCsXLlSu2348OECgAgODi7yHtRqtcjKyhJ37twRAMSff/6p/a59+/aiSpUqIi4u7oUxbd26VbstOjpamJqaiq+++qrIa9epU0e4ubkVuU9ebdq0EW3atMm3ffjw4cLb21v7e+7zqF69usjMzNTZd/v27QKA2Lt3r3Zbdna28PDwEH379tVue++994S1tbW4c+eOzvHfffedACAuXbpU7LiJjBWbpYiM0IoVK2BhYYGBAwcCAKytrfHWW2/h6NGjOqN8du3ahXbt2sHf37/Qc+3atQu1atVCx44d9Rpj3759822Li4vD2LFj4enpCVNTU5iZmcHb2xsAEBkZCUDTP+fw4cPo378/nJ2dCz1/27Zt0bBhQ/z444/abT/99BNkMhneffddvd5LSfXq1UtbhcrVtWtXuLm56VRe9uzZg/v372PUqFHabTt27EC7du3g4eGB7Oxs7adr164AgMOHDxvmJogqMCY3REbm+vXrOHLkCLp37w4hBJ48eYInT55o+4gEBwdr93348GGhfUpKsk9JWVpawtbWVmebWq1Gp06dsGXLFkydOhX79+/HqVOn8M8//wDQNLUBwOPHj6FSqYoV08SJE7F//35cuXIFWVlZWL58Ofr16wc3N7cij/Py8sLDhw+Rmppayjssmru7e75tpqamGDp0KLZu3YonT54AAFatWgV3d3d07txZu9+DBw/w119/wczMTOdTr149ACiwfxLRy4bJDZGRCQ4OhhACf/zxB+zt7bWf7t27AwBWr14NlUoFAHB2dsa9e/eKPF9x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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot training & validation loss\n",
    "plt.plot(history.history['loss'], label='Train Loss')\n",
    "plt.plot(history.history['val_loss'], label='Validation Loss')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "plt.title('Loss Curve')\n",
    "plt.show()\n",
    "\n",
    "# Plot training & validation accuracy\n",
    "plt.plot(history.history['accuracy'], label='Train Accuracy')\n",
    "plt.plot(history.history['val_accuracy'], label='Validation Accuracy')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "plt.title('Accuracy Curve')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd14a53d-ccd2-4d02-8ef7-f9d34abbf9b9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:anaconda3]",
   "language": "python",
   "name": "conda-env-anaconda3-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
