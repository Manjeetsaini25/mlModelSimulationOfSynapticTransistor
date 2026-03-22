igzo_project/
├── data/
│   ├── IDVD_IDBG_IGZO_MgO.csv        ← Experimental transfer curve data
│   └── filterchar_IGZO_MgO_new.csv   ← EPSC frequency response data
│
├── models/                           ← Trained models are stored here
│   ├── device_params.pkl             ← Extracted device parameters (Vth, ION/IOFF, SS)
│   ├── poly_model.pkl                ← Polynomial IDS predictor
│   ├── nn_model.pkl                  ← Neural Network IDS predictor
│   ├── gain_model.pkl                ← EPSC Gain prediction model
│   ├── igzo_digit_model.pkl          ← Standard digit recognition model
│   └── igzo_TRUE_digit_model.pkl     ← Physics-aware IGZO model (BEST)
│
├── 1_train_model.py                  ← STEP 1: Train device-level models
├── 2_digit_recognition.py            ← STEP 2: Standard digit recognition
├── 2b_digit_recognition_IGZO_TRUE.py ← STEP 2b: Physics-based IGZO learning (BEST)
├── 3_app.py                          ← STEP 3: GUI application
├── requirements.txt
└── README.md
