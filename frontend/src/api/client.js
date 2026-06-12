const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const SESSION_KEY = 'session_id';

function getSessionId() {
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
  }
  return sessionId;
}

function sessionHeaders(extra = {}) {
  return {
    'X-Session-Id': getSessionId(),
    ...extra,
  };
}

async function handleResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail || data.message || data.error || `Request failed (${response.status})`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data;
}

export async function getStatus() {
  const response = await fetch(`${API_BASE_URL}/status`, {
    headers: sessionHeaders(),
  });
  return handleResponse(response);
}

export async function uploadPdfs(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    headers: sessionHeaders(),
    body: formData,
  });
  return handleResponse(response);
}

export async function getUploadStatus(taskId) {
  const response = await fetch(`${API_BASE_URL}/upload/status/${taskId}`, {
    headers: sessionHeaders(),
  });
  if (response.status === 404) {
    throw new Error('Upload task lost — the server may have restarted. Please try uploading again.');
  }
  return handleResponse(response);
}

export async function pollUploadUntilComplete(
  taskId,
  { intervalMs = 750, maxAttempts = 120, onStatus } = {},
) {
  for (let i = 0; i < maxAttempts; i += 1) {
    const status = await getUploadStatus(taskId);
    onStatus?.(status);
    if (status.status === 'completed' || status.status === 'failed') {
      return status;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error('Upload processing timed out');
}

export async function sendChatMessage(question, conversationId = null) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: sessionHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      question,
      conversation_id: conversationId,
    }),
  });
  return handleResponse(response);
}

export async function streamChatMessage(question, conversationId, onToken, onDone) {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: sessionHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      question,
      conversation_id: conversationId,
    }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `Stream failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const payload = JSON.parse(line.slice(6));
      if (payload.token) onToken(payload.token);
      if (payload.done) onDone(payload.sources || []);
    }
  }
}

export async function getConversations() {
  const response = await fetch(`${API_BASE_URL}/conversations`, {
    headers: sessionHeaders(),
  });
  return handleResponse(response);
}

export async function createConversation(title = 'New conversation') {
  const response = await fetch(`${API_BASE_URL}/conversations`, {
    method: 'POST',
    headers: sessionHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title }),
  });
  return handleResponse(response);
}

export async function getConversationMessages(conversationId) {
  const response = await fetch(`${API_BASE_URL}/conversations/${conversationId}/messages`, {
    headers: sessionHeaders(),
  });
  return handleResponse(response);
}

export { API_BASE_URL };
