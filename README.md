# SignVision

The browser frontend is kept to three files: `public/index.html`, `public/styles.css`, and `public/script.js`. The Python model, dataset, and evaluation artifacts remain separate project files.

## Run locally

Install the Python dependencies, then run the app:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm install
npm run dev
```

Open `http://127.0.0.1:8000`. Camera access requires permission. Capture a frame or upload a PNG/JPEG to send it to the local Keras model. The result panel shows the returned class, confidence, top three classes, and measured inference time.

The model was trained on 28x28 grayscale Sign Language MNIST crops. It does not extract hand landmarks from live video, and its test-set metrics do not measure webcam accuracy. The dashboard labels unavailable measurements as N/A.

## Cloudflare Pages

Pages serves the static frontend only; it does not host the Python inference API. Build output is the `public` directory.

```powershell
npm run pages:dev
npx wrangler login
npm run deploy
```

## Project data

The included CNN has two convolution and max-pooling blocks, a 128-unit dense layer, and 24 static classes (A-I and K-Y). J and Z are excluded because their signs require motion. Evaluation images and results are stored under `outputs/` and copied to `public/` for the dashboard.