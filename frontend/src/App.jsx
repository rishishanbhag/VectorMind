import { useCallback, useEffect, useState } from 'react';
import {
  createConversation,
  getConversations,
  getStatus,
} from './api/client';
import PdfUpload from './components/PdfUpload';
import ChatInterface from './components/ChatInterface';
import StatusBadge from './components/StatusBadge';
import './App.css';

function BrandMark() {
  return <div className="brand-mark">PC</div>;
}

const STEPS = [
  {
    step: '1',
    title: 'Upload your PDFs',
    desc: 'Drop documents onto the upload area. We extract and index every page automatically.',
  },
  {
    step: '2',
    title: 'Build your knowledge base',
    desc: 'Content is chunked, embedded, and stored in a vector database for semantic search.',
  },
  {
    step: '3',
    title: 'Ask anything',
    desc: 'Chat with your documents. Answers are grounded in retrieved sources with citations.',
  },
];

function App() {
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);

  const refreshStatus = useCallback(async () => {
    try {
      const data = await getStatus();
      setStatus(data);
      setStatusError(null);
    } catch (err) {
      setStatusError(err.message);
    }
  }, []);

  const loadConversations = useCallback(async () => {
    try {
      const convs = await getConversations();
      setConversations(convs);
      if (convs.length && !activeConversationId) {
        setActiveConversationId(convs[0].id);
      }
    } catch {
      setConversations([]);
    }
  }, [activeConversationId]);

  useEffect(() => {
    refreshStatus();
    loadConversations();
  }, [refreshStatus, loadConversations]);

  const handleNewConversation = async () => {
    const conv = await createConversation();
    setConversations((prev) => [conv, ...prev]);
    setActiveConversationId(conv.id);
  };

  const handleUploadComplete = () => {
    refreshStatus();
    loadConversations();
  };

  const ready = status?.vectorstore_loaded;

  if (!ready) {
    return (
      <div className="app landing-page">
        <header className="landing-header">
          <div className="brand">
            <BrandMark />
            <span className="brand-name">PDF Chat</span>
          </div>
        </header>

        <main className="landing-main">
          <section className="landing-hero">
            <span className="section-label">Document intelligence, on demand</span>
            <h1>Ask questions directly from your PDFs</h1>
            <p className="subtitle">
              Upload documents, build a searchable knowledge base, and get AI-powered answers
              grounded in your content.
            </p>
          </section>

          <section className="hero-upload">
            <PdfUpload
              variant="hero"
              onUploadComplete={handleUploadComplete}
              disabled={!!statusError}
            />
          </section>

          {statusError && (
            <p className="message error landing-error">
              Cannot reach backend: {statusError}. Make sure FastAPI is running on port 8000.
            </p>
          )}

          <section className="steps-section">
            <h2>From upload to insight</h2>
            <div className="steps-grid">
              {STEPS.map((item) => (
                <article key={item.step} className="step-card">
                  <span className="step-number">{item.step}</span>
                  <h3>{item.title}</h3>
                  <p>{item.desc}</p>
                </article>
              ))}
            </div>
          </section>
        </main>

        <footer className="landing-footer">
          <span>PDF Chat · Document Q&amp;A powered by RAG</span>
        </footer>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <BrandMark />
          <div>
            <h1>PDF Chat</h1>
            <p className="subtitle">Upload documents and ask questions powered by RAG.</p>
          </div>
        </div>
        <div className="header-actions">
          <StatusBadge status={status} />
        </div>
      </header>

      {statusError && (
        <div className="banner error">
          Cannot reach backend: {statusError}. Make sure FastAPI is running on port 8000.
        </div>
      )}

      <main className="app-main">
        <aside className="sidebar">
          <PdfUpload onUploadComplete={handleUploadComplete} disabled={!!statusError} />

          <div className="conversations-panel">
            <div className="conversations-header">
              <h3>Conversations</h3>
              <button type="button" className="btn secondary small" onClick={handleNewConversation}>
                New
              </button>
            </div>
            <ul className="conversation-list">
              {conversations.map((conv) => (
                <li key={conv.id}>
                  <button
                    type="button"
                    className={conv.id === activeConversationId ? 'active' : ''}
                    onClick={() => setActiveConversationId(conv.id)}
                  >
                    {conv.title}
                  </button>
                </li>
              ))}
            </ul>
          </div>

          {status && (
            <div className="status-details">
              <h3>System Status</h3>
              <ul>
                {Object.entries(status).map(([key, value]) => (
                  <li key={key}>
                    <span>{key.replace(/_/g, ' ')}</span>
                    <span className={value ? 'ok' : 'no'}>{String(value)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </aside>

        <section className="chat-section">
          <ChatInterface
            ready={status?.vectorstore_loaded}
            conversationId={activeConversationId}
            onConversationChange={setActiveConversationId}
          />
        </section>
      </main>
    </div>
  );
}

export default App;
