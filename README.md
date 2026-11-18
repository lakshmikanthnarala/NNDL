# SignSpeak

SignSpeak is a small ASL alphabet project with a static website and an optional
Python model-training pipeline.

## Website

The Cloudflare Pages site is plain HTML, CSS, and JavaScript in `public/`. It
provides a responsive project overview, camera/image preview, manual alphabet
text builder, and saved training metrics. It does **not** run the Python model or
recognize signs. The original model and training scripts remain available for
experimentation, but are not part of the deployed site.

### Run locally

On Windows, double-click `run_website.bat`, then open
`http://localhost:8000`. Or run from the project directory:

```powershell
python -m http.server 8000 --directory public
```

For Cloudflare Pages development, install the Wrangler development dependency
and start the local preview:

```powershell
npm install
npm run dev
```

### Deploy to Cloudflare Pages

Connect the repository to Cloudflare Pages with these build settings:

- Framework preset: **None**
- Build command: leave blank
- Build output directory: `public`

Alternatively, authenticate with Wrangler and deploy from the project root:

```powershell
npm install
npx wrangler login
npm run deploy
```

The Pages build contains only the static website and its metrics JSON. It does
not require Docker, a Python runtime, or a server-side function. Camera preview
requires browser permission and a secure context; `localhost` and deployed
HTTPS sites are supported.

## Optional model training

The model is a 24-class CNN trained on Sign Language MNIST. J and Z are
excluded because they require movement. The tracked `dataset/` CSVs are used
for training and held-out evaluation.

Use Python 3.10–3.11 for the TensorFlow compatibility range:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python -m unittest discover -s tests -v
```

Training saves the Keras model and class mapping to `models/`, evaluation
metrics and plots to `outputs/`, and refreshes the metrics shown on the website
in `public/data/evaluation_results.json`.

The CNN takes 28 × 28 grayscale images, uses convolution/max-pooling blocks
with a 128-unit dense layer, and outputs 24 letter classes. The reported
training and test metrics are measurements from the included training run,
not a guarantee of accuracy on real-world images.

## Project structure

```text
public/              Static Cloudflare Pages website
  data/              Metrics displayed by the site
dataset/             Sign Language MNIST CSV files
models/              Trained Keras model and label mapping
outputs/             Training metrics, reports, and plots
tests/               Python model and preprocessing tests
train_model.py       Optional CNN training and evaluation
predict.py           Python image prediction helper
preprocessing.py     Python image preprocessing
requirements.txt     Optional Python training dependencies
```

This is an educational static alphabet project, not a continuous sign-language
translator. It does not interpret motion, grammar, facial expression, or
sentence meaning.
