const letters = "ABCDEFGHI KLMNOPQRSTUVWXY".replaceAll(" ", "");
const alphabetGrid = document.querySelector("#alphabet-grid");
const textArea = document.querySelector("#built-text");
const imageFile = document.querySelector("#image-file");
const imagePreview = document.querySelector("#image-preview");
const emptyPreview = document.querySelector("#empty-preview");
const video = document.querySelector("#camera-video");
const canvas = document.querySelector("#capture-canvas");
const imageStatus = document.querySelector("#image-status");
const copyStatus = document.querySelector("#copy-status");
const predictionResult = document.querySelector("#prediction-result");
const predictedLetter = document.querySelector("#predicted-letter");
const predictionConfidence = document.querySelector("#prediction-confidence");
const addPrediction = document.querySelector("#add-prediction");
const serviceStatus = document.querySelector("#service-status");
let cameraStream;
let previewUrl;
let predictionRequest = 0;
let currentPrediction;

for (const letter of letters) {
  const button = document.createElement("button");
  button.className = "letter-button";
  button.type = "button";
  button.textContent = letter;
  button.setAttribute("aria-label", `Add letter ${letter}`);
  button.addEventListener("click", () => insertText(letter));
  alphabetGrid.append(button);
}

function insertText(value) {
  const start = textArea.selectionStart;
  const end = textArea.selectionEnd;
  textArea.setRangeText(value, start, end, "end");
  textArea.focus();
  updateCharacterCount();
}

function updateCharacterCount() {
  const count = textArea.value.length;
  document.querySelector("#character-count").textContent =
    `${count} ${count === 1 ? "character" : "characters"}`;
}

function showPreview(source, altText) {
  stopCamera();
  imagePreview.src = source;
  imagePreview.alt = altText;
  imagePreview.hidden = false;
  video.hidden = true;
  emptyPreview.hidden = true;
}

function stopCamera() {
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) track.stop();
    cameraStream = undefined;
  }
  video.srcObject = null;
  video.hidden = true;
  document.querySelector("#camera-capture").disabled = true;
  document.querySelector("#camera-stop").disabled = true;
}

imageFile.addEventListener("change", () => {
  const file = imageFile.files?.[0];
  if (!file) return;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  showPreview(previewUrl, "Selected sign image preview");
  recognizeImage(previewUrl, file.name);
});

document.querySelector("#camera-start").addEventListener("click", async () => {
  predictionRequest += 1;
  currentPrediction = undefined;
  predictionResult.hidden = true;
  addPrediction.disabled = true;
  if (!navigator.mediaDevices?.getUserMedia) {
    imageStatus.textContent = "Camera access is not available in this browser or page context.";
    return;
  }
  try {
    stopCamera();
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = cameraStream;
    video.hidden = false;
    imagePreview.hidden = true;
    emptyPreview.hidden = true;
    document.querySelector("#camera-capture").disabled = false;
    document.querySelector("#camera-stop").disabled = false;
    imageStatus.textContent = "Camera is on. Take a snapshot to keep a still preview.";
  } catch (error) {
    imageStatus.textContent =
      error.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access in your browser settings to try again."
        : "Could not start the camera. Check that a camera is connected and available.";
  }
});

document.querySelector("#camera-capture").addEventListener("click", () => {
  if (!video.videoWidth || !video.videoHeight) {
    imageStatus.textContent = "The camera is still starting. Try taking a snapshot in a moment.";
    return;
  }
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = undefined;
  }
  showPreview(canvas.toDataURL("image/png"), "Camera snapshot preview");
  recognizeImage(imagePreview.src, "camera snapshot");
});

document.querySelector("#camera-stop").addEventListener("click", () => {
  predictionRequest += 1;
  currentPrediction = undefined;
  predictionResult.hidden = true;
  addPrediction.disabled = true;
  stopCamera();
  imagePreview.hidden = true;
  emptyPreview.hidden = false;
  imageStatus.textContent = "Camera stopped.";
});

addPrediction.addEventListener("click", () => {
  if (!currentPrediction) return;
  insertText(currentPrediction);
  imageStatus.textContent = `Added ${currentPrediction} to your message.`;
});

document.querySelector("#add-space").addEventListener("click", () => insertText(" "));
document.querySelector("#delete-last").addEventListener("click", () => {
  const start = textArea.selectionStart;
  const end = textArea.selectionEnd;
  if (start !== end) {
    textArea.setRangeText("", start, end, "end");
  } else if (start > 0) {
    textArea.setRangeText("", start - 1, start, "end");
  }
  textArea.focus();
  updateCharacterCount();
});
document.querySelector("#clear-text").addEventListener("click", () => {
  textArea.value = "";
  updateCharacterCount();
  textArea.focus();
});
textArea.addEventListener("input", updateCharacterCount);
document.querySelector("#copy-text").addEventListener("click", async () => {
  if (!textArea.value) {
    copyStatus.textContent = "Add some letters before copying.";
    return;
  }
  try {
    await navigator.clipboard.writeText(textArea.value);
    copyStatus.textContent = "Message copied.";
  } catch {
    textArea.focus();
    textArea.select();
    copyStatus.textContent = "Clipboard access is unavailable. Select and copy the message.";
  }
});
document.querySelector("#download-text").addEventListener("click", () => {
  if (!textArea.value) {
    copyStatus.textContent = "Add some letters before downloading.";
    return;
  }
  const file = new Blob([textArea.value], { type: "text/plain;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(file);
  link.download = "signspeak-message.txt";
  link.click();
  URL.revokeObjectURL(link.href);
  copyStatus.textContent = "Message downloaded.";
});

async function recognizeImage(source, description) {
  const requestId = ++predictionRequest;
  currentPrediction = undefined;
  predictionResult.hidden = true;
  addPrediction.disabled = true;
  imageStatus.textContent = `Preparing ${description} for the local model…`;
  try {
    const image = new Image();
    image.src = source;
    await image.decode();
    const scale = Math.min(1, 1024 / Math.max(image.naturalWidth, image.naturalHeight));
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not prepare this image for prediction.");
    context.drawImage(image, 0, 0, canvas.width, canvas.height);

    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.9) }),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(result.error || `Local prediction service returned ${response.status}.`);
    }
    if (!letters.includes(result.letter) || !Number.isFinite(result.confidence)) {
      throw new Error("The local model returned an invalid prediction.");
    }
    if (requestId !== predictionRequest) return;

    currentPrediction = result.letter;
    predictedLetter.textContent = result.letter;
    predictionConfidence.textContent = `${(result.confidence * 100).toFixed(1)}% confidence`;
    predictionResult.hidden = false;
    addPrediction.disabled = false;
    imageStatus.textContent = "Prediction complete. Add the letter to your message or try another image.";
  } catch (error) {
    if (requestId !== predictionRequest) return;
    imageStatus.textContent = `Could not get a prediction: ${error.message}`;
  }
}

async function loadServiceStatus() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error(`Local service unavailable (${response.status}).`);
    const health = await response.json();
    if (!health.model) throw new Error("Local model is not loaded.");
    serviceStatus.lastChild.textContent = " MODEL ONLINE";
    serviceStatus.classList.remove("offline");
  } catch {
    serviceStatus.lastChild.textContent = " MODEL OFFLINE";
    serviceStatus.classList.add("offline");
  }
}

async function loadMetrics() {
  const container = document.querySelector("#metrics");
  try {
    const response = await fetch("/data/evaluation_results.json");
    if (!response.ok) throw new Error(`Could not load project results (${response.status}).`);
    const metrics = await response.json();
    const values = [
      ["training_accuracy", "Training accuracy", true],
      ["testing_accuracy", "Held-out test accuracy", true],
      ["number_of_classes", "Static classes", false],
    ];
    container.replaceChildren(...values.map(([key, label, percent]) => {
      const card = document.createElement("div");
      card.className = "metric";
      const value = document.createElement("strong");
      value.textContent = percent ? `${(metrics[key] * 100).toFixed(1)}%` : metrics[key];
      const caption = document.createElement("span");
      caption.textContent = label;
      card.append(value, caption);
      return card;
    }));
  } catch (error) {
    container.textContent = error.message;
    container.classList.add("feedback");
  }
}

window.addEventListener("pagehide", stopCamera);
updateCharacterCount();
loadServiceStatus();
loadMetrics();
