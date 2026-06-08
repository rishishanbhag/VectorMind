import { useEffect, useState } from 'react';
import {
  getConversationMessages,
  sendChatMessage,
  streamChatMessage,
} from '../api/client';

function MessageContent({ content }) {
  if (!content) return null;

  const blocks = [];
  let listItems = [];

  const flushList = () => {
    if (listItems.length) {
      blocks.push(
        <ul key={`list-${blocks.length}`}>
          {listItems.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>,
      );
      listItems = [];
    }
  };

  content.split('\n').forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      return;
    }
    if (trimmed.startsWith('### ')) {
      flushList();
      blocks.push(<h4 key={index}>{trimmed.slice(4)}</h4>);
      return;
    }
    if (trimmed.startsWith('## ')) {
      flushList();
      blocks.push(<h3 key={index}>{trimmed.slice(3)}</h3>);
      return;
    }
    if (trimmed.startsWith('- ')) {
      listItems.push(trimmed.slice(2));
      return;
    }
    flushList();
    blocks.push(<p key={index}>{trimmed}</p>);
  });

  flushList();

  return <div className="message-content">{blocks}</div>;
}

function SourceCards({ sources }) {
  if (!sources?.length) return null;

  return (
    <details className="source-cards">
      <summary>Sources ({sources.length})</summary>
      <ul>
        {sources.map((src, i) => (
          <li key={i}>
            <strong>
              {src.metadata?.filename || 'Document'} — page {src.metadata?.page ?? '?'}
            </strong>
            <p>{src.content?.slice(0, 280)}{src.content?.length > 280 ? '…' : ''}</p>
          </li>
        ))}
      </ul>
    </details>
  );
}

export default function ChatInterface({ ready, conversationId, onConversationChange }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [useStreaming, setUseStreaming] = useState(true);
  const [activeConversationId, setActiveConversationId] = useState(conversationId);

  useEffect(() => {
    setActiveConversationId(conversationId);
  }, [conversationId]);

  useEffect(() => {
    if (!activeConversationId) {
      setMessages([]);
      return;
    }

    getConversationMessages(activeConversationId)
      .then((msgs) => setMessages(msgs.map((m) => ({ role: m.role, content: m.content, sources: m.sources }))))
      .catch(() => setMessages([]));
  }, [activeConversationId]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    if (!ready) {
      setError('Please upload and process PDF documents before chatting.');
      return;
    }

    setError(null);
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    setLoading(true);

    try {
      if (useStreaming) {
        let streamed = '';
        setMessages((prev) => [...prev, { role: 'assistant', content: '', sources: [] }]);

        await streamChatMessage(
          question,
          activeConversationId,
          (token) => {
            streamed += token;
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: 'assistant', content: streamed, sources: [] };
              return updated;
            });
          },
          (sources) => {
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: 'assistant', content: streamed, sources };
              return updated;
            });
          },
        );
      } else {
        const result = await sendChatMessage(question, activeConversationId);
        if (result.conversation_id && result.conversation_id !== activeConversationId) {
          setActiveConversationId(result.conversation_id);
          onConversationChange?.(result.conversation_id);
        }
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: result.answer, sources: result.sources || [] },
        ]);
      }
    } catch (err) {
      setError(err.message);
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === 'assistant' && !last.content) {
          return prev.slice(0, -1);
        }
        return prev.slice(0, -1);
      });
      setInput(question);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <h2>Chat</h2>
        <p className="panel-desc">Ask questions about your uploaded documents.</p>
        <label className="stream-toggle">
          <input
            type="checkbox"
            checked={useStreaming}
            onChange={(e) => setUseStreaming(e.target.checked)}
          />
          Stream responses
        </label>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>No messages yet. Upload PDFs, then ask a question.</p>
          </div>
        )}
        {messages.map((msg, index) => (
          <div key={index} className={`chat-bubble ${msg.role}`}>
            <span className="bubble-label">{msg.role === 'user' ? 'You' : 'Assistant'}</span>
            <MessageContent content={msg.content} />
            {msg.role === 'assistant' && <SourceCards sources={msg.sources} />}
          </div>
        ))}
        {loading && !useStreaming && (
          <div className="chat-bubble assistant loading">
            <span className="bubble-label">Assistant</span>
            <p>Thinking…</p>
          </div>
        )}
      </div>

      {error && <p className="message error chat-error">{error}</p>}

      <form className="chat-input-row" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={ready ? 'Ask a question about your documents…' : 'Upload documents to start chatting'}
          disabled={loading || !ready}
        />
        <button type="submit" className="btn primary" disabled={loading || !ready || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
