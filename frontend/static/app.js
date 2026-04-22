const FIELDS = [
  "gender","SeniorCitizen","Partner","Dependents","tenure",
  "PhoneService","MultipleLines","InternetService","OnlineSecurity",
  "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
  "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
  "MonthlyCharges","TotalCharges"
];

const NUM_FIELDS = new Set(["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]);

const GAUGE_CIRCUMFERENCE = Math.PI * 50; // r=50 → ~157.08

async function checkHealth() {
  const dot   = document.getElementById("status-dot");
  const label = document.getElementById("status-label");
  try {
    const r = await fetch("/health");
    const d = await r.json();
    if (d.status === "ok" && d.model_loaded) {
      dot.className     = "status-dot online";
      label.textContent = "API Online";
    } else {
      dot.className     = "status-dot offline";
      label.textContent = "Model Yüklenmedi";
    }
  } catch {
    dot.className     = "status-dot offline";
    label.textContent = "API Offline";
  }
}

function buildPayload() {
  const payload = {};
  for (const f of FIELDS) {
    const el = document.getElementById(f);
    if (!el) continue;
    payload[f] = NUM_FIELDS.has(f) ? Number(el.value) : el.value;
  }
  return payload;
}

function setLoading(on) {
  const btn     = document.getElementById("predict-btn");
  const spinner = document.getElementById("spinner");
  btn.disabled          = on;
  spinner.style.display = on ? "block" : "none";
}

function showError(msg) {
  const el = document.getElementById("error-msg");
  el.textContent    = msg;
  el.style.display  = "block";
  document.getElementById("result-content").style.display     = "none";
  document.getElementById("result-placeholder").style.display = "none";
}

function updateGauge(pct) {
  const arc = document.getElementById("gauge-arc");
  arc.style.strokeDashoffset = GAUGE_CIRCUMFERENCE * (1 - pct);
  arc.style.stroke = pct >= 0.65 ? "#ef4444"
                   : pct >= 0.45 ? "#f97316"
                   : "#22c55e";
}

function showResult(data) {
  document.getElementById("error-msg").style.display       = "none";
  document.getElementById("result-placeholder").style.display = "none";

  const pct     = data.churn_probability;
  const pctDisp = Math.round(pct * 100);
  const isChurn = data.churn_prediction;

  const probEl    = document.getElementById("prob-number");
  const riskEl    = document.getElementById("gauge-risk-text");
  const verdictEl = document.getElementById("verdict");
  const metaEl    = document.getElementById("meta-info");
  const contentEl = document.getElementById("result-content");

  probEl.textContent = `${pctDisp}%`;
  probEl.style.color = isChurn ? "#f87171" : "#4ade80";

  updateGauge(pct);

  riskEl.textContent = pct >= 0.65 ? "Yüksek Risk"
                     : pct >= 0.45 ? "Orta Risk"
                     : "Düşük Risk";

  verdictEl.textContent = isChurn ? "⚠ Churn Riski Var" : "✓ Müşteri Güvende";
  verdictEl.className   = `verdict ${isChurn ? "churn" : "safe"}`;

  metaEl.textContent = `Eşik: ${data.threshold_used.toFixed(2)}  ·  Olasılık: ${data.churn_probability.toFixed(4)}`;

  contentEl.style.display = "flex";
}

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  setLoading(true);
  document.getElementById("error-msg").style.display = "none";

  try {
    const resp = await fetch("/api/v1/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(buildPayload()),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || "Bilinmeyen hata");
    }
    showResult(await resp.json());
  } catch (err) {
    showError(`Hata: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

checkHealth();
