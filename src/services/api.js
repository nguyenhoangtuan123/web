export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export const classifyECG = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(`${API_URL}/classify`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) throw new Error('Classification failed');
  return response.json();
};