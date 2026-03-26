const payloadEl = document.getElementById('payload');
const resultEl = document.getElementById('result');
const signalsEl = document.getElementById('signals');
payloadEl.value = JSON.stringify(window.defaultPayload, null, 2);

function renderPrediction(data) {
  resultEl.textContent = JSON.stringify({
    fraud_probability: data.fraud_probability,
    predicted_label: data.predicted_label,
    risk_level: data.risk_level
  }, null, 2);

  signalsEl.innerHTML = '';
  (data.shap_top_features || []).forEach(item => {
    const row = document.createElement('div');
    row.className = 'signal';
    row.innerHTML = `<strong>${item.feature}</strong><span>value=${Number(item.value).toFixed(4)} | shap=${Number(item.shap_value).toFixed(6)}</span>`;
    signalsEl.appendChild(row);
  });
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(data));
  return data;
}

document.getElementById('loadSampleBtn').addEventListener('click', async () => {
  const response = await fetch('/sample');
  const data = await response.json();
  payloadEl.value = JSON.stringify(data.payload, null, 2);
});

document.getElementById('scoreBtn').addEventListener('click', async () => {
  try {
    const payload = JSON.parse(payloadEl.value);
    const data = await postJson('/predict', payload);
    renderPrediction(data);
  } catch (err) {
    resultEl.textContent = `Error: ${err.message}`;
  }
});

document.getElementById('scoreCsvBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('csvFile');
  if (!fileInput.files.length) {
    resultEl.textContent = 'Please choose a CSV file first.';
    return;
  }
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('row_index', document.getElementById('rowIndex').value || '0');
  const response = await fetch('/predict-csv', { method: 'POST', body: formData });
  const data = await response.json();
  if (!response.ok) {
    resultEl.textContent = `Error: ${JSON.stringify(data)}`;
    return;
  }
  renderPrediction(data);
});
