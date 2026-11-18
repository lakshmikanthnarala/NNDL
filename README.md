# SignSpeak

SignSpeak is an ASL alphabet project with a browser interface connected to a
local Python prediction service and model-training pipeline.

## Run locally

On Windows, double-click `run_website.bat`, then open
`http://127.0.0.1:8000`. The launcher uses the project virtual environment
when available. To set up the Python environment manually:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python server.py
```

The server loads the included model and serves both the website and prediction
API on the same local port. Selecting an image or taking a camera snapshot
sends it to the model on this computer; predictions can be added directly to
the text builder. Images are not uploaded to a remote server. To run through
npm, activate the Python environment first and use `npm run dev`.

The model is a 24-class CNN trained on Sign Language MNIST. J and Z are
excluded because they require movement. Predictions on ordinary photos or
camera images are experimental: the training images are standardized grayscale
hand crops, so held-out dataset accuracy does not guarantee real-world
recognition accuracy. Camera access requires browser permission.

## Cloudflare Pages preview/deployment

Cloudflare Pages deploys the static files only and does not include the Python
prediction service. Use the local app above for model predictions. To preview
the static Pages build:

```powershell
npm install
npm run pages:dev
```

To deploy the static Pages build:

```powershell
npm install
npx wrangler login
npm run deploy
```

Configure Pages with framework preset **None**, no build command, and `public`
as the build output directory.

## Model training

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
public/              Browser interface, metrics, and styles
  data/              Metrics displayed by the site
dataset/             Sign Language MNIST CSV files
models/              Trained Keras model and label mapping
outputs/             Training metrics, reports, and plots
tests/               Python model and preprocessing tests
server.py            Local website server and prediction API
train_model.py       Optional CNN training and evaluation
predict.py           Python image prediction helper
preprocessing.py     Python image preprocessing
requirements.txt     Python model and training dependencies
```

This is an educational static alphabet project, not a continuous sign-language
translator. It does not interpret motion, grammar, facial expression, or
sentence meaning.
