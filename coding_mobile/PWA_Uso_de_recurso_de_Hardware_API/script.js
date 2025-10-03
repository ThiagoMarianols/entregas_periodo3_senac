const btnLocal = document.getElementById("btn-local");
const btnPhoto = document.getElementById("btn-photo");
const btnCapture = document.getElementById("btn-capture");
const locationEl = document.getElementById("location");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const photo = document.getElementById("photo");
const historyDiv = document.getElementById("history");

let currentLocation = "";
let activeStream = null; // mantém a referência para parar a câmera após capturar

// PEGAR LOCALIZAÇÃO
if (btnLocal) {
  btnLocal.addEventListener("click", () => {
    if (!navigator.geolocation) {
      alert("Geolocalização não suportada!");
      return;
    }

    // mostra que tá carregando a localização
    if (locationEl) {
      locationEl.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Obtendo localização...`;
    }
    btnLocal.disabled = true;

    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        try {
          const { latitude, longitude } = pos.coords;

          // Chama API do OpenStreetMap (Nominatim)
          const res = await fetch(
            `https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json`
          );
          const data = await res.json();
          const addr = data.address || {};
          const formatted = formatAddress(addr);
          currentLocation = formatted || data.display_name;
          if (locationEl) locationEl.textContent = currentLocation;
        } catch (e) {
          if (locationEl) locationEl.textContent = "Não foi possível obter o endereço.";
          console.error(e);
        } finally {
          btnLocal.disabled = false;
        }
      },
      (err) => {
        if (locationEl) locationEl.textContent = "Permissão negada ou indisponível.";
        console.error(err);
        btnLocal.disabled = false;
      }
    );
  });
}

// Formata endereço no padrão: Bairro, Cidade, Estado, País
function formatAddress(addr) {
  const neighborhood = addr.suburb || addr.neighbourhood || addr.quarter || addr.hamlet || addr.locality;
  const city = addr.city || addr.town || addr.village || addr.municipality || addr.county; // fallback amplo
  const state = addr.state;
  const country = addr.country;
  const parts = [neighborhood, city, state, country].filter(Boolean);
  return parts.join(", ");
}

// --- INICIAR CÂMERA ---
if (btnPhoto) {
  btnPhoto.addEventListener("click", async () => {
    try {
      activeStream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (video) {
        video.classList.remove("d-none");
        video.srcObject = activeStream;
      }
      if (btnCapture) btnCapture.classList.remove("d-none");
      // opcional: desabilita o botão de iniciar enquanto a câmera está ativa
      btnPhoto.disabled = true;
    } catch (err) {
      alert("Não foi possível acessar a câmera");
      console.error(err);
    }
  });
}

// --- CAPTURAR FOTO NO MOMENTO DESEJADO ---
if (btnCapture) {
  btnCapture.addEventListener("click", () => {
    if (!canvas || !video) return;
    if (!activeStream) return;

    canvas.classList.remove("d-none");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const imgData = canvas.toDataURL("image/png");
    if (photo) photo.src = imgData;

    // parar e ocultar câmera
    activeStream.getTracks().forEach(track => track.stop());
    activeStream = null;
    if (video) video.classList.add("d-none");
    if (btnCapture) btnCapture.classList.add("d-none");
    if (btnPhoto) btnPhoto.disabled = false;

    // salvar e atualizar galeria
    saveMoment(imgData, currentLocation);
    loadHistory();

    // limpar prévia para permitir nova captura
    try {
      if (photo) {
        photo.removeAttribute("src");
      }
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvas.classList.add("d-none");
      }
    } catch (e) {
      console.warn("Falha ao limpar prévia:", e);
    }
  });
}

// --- SALVAR HISTÓRICO ---
function saveMoment(photo, location) {
  const momentos = JSON.parse(localStorage.getItem("momentos") || "[]");
  momentos.push({
    photo,
    location,
    date: new Date().toLocaleString()
  });
  localStorage.setItem("momentos", JSON.stringify(momentos));
}

// --- CARREGAR HISTÓRICO ---
function loadHistory() {
  if (!historyDiv) return; // só renderiza quando existir o container (Galeria.html)
  const momentos = JSON.parse(localStorage.getItem("momentos") || "[]");
  let html = "";
  if (momentos.length > 0) {
    html += `
      <div class="d-flex justify-content-end mb-3">
        <button class="btn btn-outline-danger btn-sm" id="btn-clear-all">Apagar todas</button>
      </div>
    `;
  }
  momentos.forEach((m, i) => {
    html += `
      <div class="col-12 col-md-4 mb-3">
        <div class="card">
          <img src="${m.photo}" class="card-img-top"/>
          <div class="card-body">
            <p class="card-text"><strong>Local:</strong> ${m.location || "Não informado"}</p>
            <p class="card-text"><small>${m.date}</small></p>
            <button class="btn btn-sm btn-outline-danger w-100 mt-2 btn-delete" data-index="${i}">Apagar</button>
          </div>
        </div>
      </div>
    `;
  });
  historyDiv.innerHTML = html;
}

// Atualiza a galeria automaticamente quando o localStorage mudar (ex.: foto salva no index)
window.addEventListener("storage", (e) => {
  if (e.key === "momentos") {
    loadHistory();
  }
});

// carrega histórico ao abrir (apenas na Galeria)
if (historyDiv) {
  loadHistory();
}

// --- AÇÕES DA GALERIA (delegação de eventos) ---
if (historyDiv) {
  historyDiv.addEventListener("click", (ev) => {
    const delBtn = ev.target.closest && ev.target.closest(".btn-delete");
    if (delBtn) {
      const idx = Number(delBtn.getAttribute("data-index"));
      if (!Number.isNaN(idx)) {
        const ok = confirm("Apagar esta foto?");
        if (ok) deleteMomentAt(idx);
      }
      return;
    }

    const clearAllBtn = ev.target.closest && ev.target.closest("#btn-clear-all");
    if (clearAllBtn) {
      const ok = confirm("Apagar todas as fotos?");
      if (ok) clearAllMoments();
      return;
    }
  });
}

function deleteMomentAt(index) {
  const momentos = JSON.parse(localStorage.getItem("momentos") || "[]");
  if (index < 0 || index >= momentos.length) return;
  momentos.splice(index, 1);
  localStorage.setItem("momentos", JSON.stringify(momentos));
  loadHistory();
}

function clearAllMoments() {
  localStorage.removeItem("momentos");
  loadHistory();
}
