import CONFIG from '../config.js';

// api.js
const ENDPOINTS = {
  PREDICT: `${CONFIG.BASE_URL}/predict`,
};

export async function predictPest(imageBlob) {
  const formData = new FormData();
  formData.append('file', imageBlob);

  const response = await fetch(ENDPOINTS.PREDICT, {
    method: 'POST',
    body: formData,
  });

  return await response.json();
}