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
let cameraStream;
let previewUrl;

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
  imageStatus.textContent = `Previewing ${file.name}. No image recognition is performed.`;
});

document.querySelector("#camera-start").addEventListener("click", async () => {
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
  imageStatus.textContent = "Snapshot saved in this page for preview only. No recognition is performed.";
});

document.querySelector("#camera-stop").addEventListener("click", () => {
  stopCamera();
  imagePreview.hidden = true;
  emptyPreview.hidden = false;
  imageStatus.textContent = "Camera stopped.";
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
loadMetrics();
