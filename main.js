const video = document.getElementById("videoPlayer");
const alertBox = document.getElementById("alertBox");
const anomalyList = document.getElementById("anomalyList");
const fileInput = document.getElementById("fileInput");

alertBox.style.display = "none";
let anomalies = [];

// Load anomalies from JSON
async function fetchAnomalies() {
  try {
    const response = await fetch("output/anomalies.json");
    anomalies = await response.json();
  } catch (error) {
    console.error("Error loading anomalies.json:", error);
  }
}

// Show alert when anomaly time is reached
function triggerAlert(timestamp) {
  const match = anomalies.find(a => {
    const anomalyTime = a.frame / 30; // Assuming 30 FPS
    return Math.floor(anomalyTime) === Math.floor(timestamp);
  });

  if (match) {
    alertBox.style.display = "block";

    setTimeout(() => {
      alertBox.style.display = "none";
    }, 2000);

    const li = document.createElement("li");
    li.textContent = `🔹 ${match.type.toUpperCase()} anomaly at ${timestamp.toFixed(2)}s (ID: ${match.id || match.id_pair})`;
    anomalyList.appendChild(li);
  }
}

// Play alerts during video playback
video.addEventListener("timeupdate", () => {
  triggerAlert(video.currentTime);
});

// User selects a new video
fileInput.addEventListener("change", function () {
  const file = this.files[0];

  if (file) {
    const url = URL.createObjectURL(file);
    video.src = url;
    video.load();

    anomalyList.innerHTML = "";
    alertBox.style.display = "none";
  }
});

// Load anomalies initially
fetchAnomalies();
