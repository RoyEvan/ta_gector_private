<!-- START API FOR MODEL HOSTING -->
conda activate train_gector37
d:
cd Kuliah\Tugas_Akhir\TrialCode\gector_grammarly\gector
uvicorn gector_infer_api:app --host 0.0.0.0 --port 8001

<!-- START BACKEND DEV -->
fastapi dev main.py

<!-- START BACKEND -->
fastapi run main.py

<!-- START FRONTEND -->
