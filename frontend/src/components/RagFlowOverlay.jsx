import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

const STAGE_ORDER = ['uploading', 'parsing', 'chunking', 'embedding', 'storing', 'complete'];
const MIN_DWELL_MS = 900;
const CHUNK_COUNT = 5;

const STAGE_LABELS = {
  uploading: 'Uploading documents…',
  parsing: 'Extracting text from PDFs…',
  chunking: 'Splitting into chunks…',
  embedding: 'Generating embeddings…',
  storing: 'Storing in vector database…',
  complete: 'Pipeline complete',
  failed: 'Processing failed',
};

const NODE_KEYS = ['document', 'chunks', 'embeddings', 'vectordb'];

function activeNodeIndex(displayStage) {
  if (displayStage === 'uploading' || displayStage === 'parsing') return 0;
  if (displayStage === 'chunking') return 1;
  if (displayStage === 'embedding') return 2;
  if (displayStage === 'storing') return 3;
  if (displayStage === 'complete') return 4;
  return 0;
}

function normalizeStage(stage) {
  if (!stage || stage === 'queued') return 'parsing';
  if (stage === 'pending') return 'parsing';
  if (stage === 'processing' && STAGE_ORDER.indexOf(stage) < 0) return 'parsing';
  return stage;
}

function stageIndex(stage) {
  const normalized = normalizeStage(stage);
  const idx = STAGE_ORDER.indexOf(normalized);
  return idx >= 0 ? idx : 0;
}

function nodeState(nodeKey, displayStage, isFailed) {
  const nodeIdx = NODE_KEYS.indexOf(nodeKey);
  const activeIdx = activeNodeIndex(displayStage);

  if (isFailed) {
    if (nodeIdx < activeIdx) return 'done';
    if (nodeIdx === activeIdx) return 'failed';
    return 'pending';
  }

  if (displayStage === 'complete') return 'done';
  if (nodeIdx < activeIdx) return 'done';
  if (nodeIdx === activeIdx) return 'active';
  return 'pending';
}

function connectorState(fromNode, toNode, displayStage, isFailed) {
  if (isFailed) return 'idle';

  const fromIdx = NODE_KEYS.indexOf(fromNode);
  const toIdx = NODE_KEYS.indexOf(toNode);
  const activeIdx = activeNodeIndex(displayStage);

  if (displayStage === 'complete') return 'done';
  if (activeIdx === toIdx) return 'active';
  if (activeIdx > fromIdx && activeIdx <= toIdx) return 'active';
  if (activeIdx > toIdx) return 'done';
  return 'idle';
}

function truncateName(name, max = 18) {
  if (!name) return '';
  return name.length > max ? `${name.slice(0, max - 1)}…` : name;
}

export default function RagFlowOverlay({
  open,
  stage,
  files = [],
  filesDone = 0,
  fileCount = 0,
  result = null,
  error = null,
  onClose,
  onDismiss,
}) {
  const [displayStage, setDisplayStage] = useState('uploading');
  const [chunkLitCount, setChunkLitCount] = useState(0);
  const targetIdxRef = useRef(0);
  const lastAdvanceRef = useRef(Date.now());
  const onDismissRef = useRef(onDismiss);
  const isFailed = Boolean(error) || stage === 'failed';

  onDismissRef.current = onDismiss;

  useEffect(() => {
    if (!open) {
      setDisplayStage('uploading');
      targetIdxRef.current = 0;
      lastAdvanceRef.current = Date.now();
      return;
    }

    if (isFailed) {
      const failedAt = normalizeStage(stage);
      setDisplayStage(failedAt === 'complete' ? 'storing' : failedAt);
      return;
    }

    const incoming = stage === 'complete' ? 'complete' : normalizeStage(stage);
    const incomingIdx = stageIndex(incoming);
    if (incomingIdx > targetIdxRef.current) {
      targetIdxRef.current = incomingIdx;
    }
  }, [open, stage, isFailed]);

  useEffect(() => {
    if (!open || isFailed) return undefined;

    const tick = () => {
      const currentIdx = stageIndex(displayStage);
      const targetIdx = targetIdxRef.current;

      if (currentIdx < targetIdx) {
        const elapsed = Date.now() - lastAdvanceRef.current;
        if (elapsed >= MIN_DWELL_MS) {
          setDisplayStage(STAGE_ORDER[currentIdx + 1]);
          lastAdvanceRef.current = Date.now();
        }
      }
    };

    const interval = setInterval(tick, 100);
    return () => clearInterval(interval);
  }, [open, displayStage, isFailed]);

  useEffect(() => {
    if (!open || isFailed || displayStage !== 'complete' || !result) return undefined;
    const timer = setTimeout(() => onDismissRef.current?.(), 1500);
    return () => clearTimeout(timer);
  }, [open, isFailed, displayStage, result]);

  useEffect(() => {
    if (!open) {
      setChunkLitCount(0);
      return undefined;
    }

    if (displayStage === 'complete' || stageIndex(displayStage) > stageIndex('chunking')) {
      setChunkLitCount(CHUNK_COUNT);
      return undefined;
    }

    if (displayStage !== 'chunking') {
      setChunkLitCount(0);
      return undefined;
    }

    setChunkLitCount(0);
    const interval = setInterval(() => {
      setChunkLitCount((count) => Math.min(count + 1, CHUNK_COUNT));
    }, 220);
    return () => clearInterval(interval);
  }, [open, displayStage]);

  if (!open) return null;

  const fileNames = files.map((f) => (typeof f === 'string' ? f : f.name));
  const primaryFile = fileNames[0] || 'document.pdf';
  const parsingLabel =
    fileCount > 1 ? `Extracting text (${filesDone || 0}/${fileCount})…` : STAGE_LABELS.parsing;

  const statusLabel = isFailed
    ? error || STAGE_LABELS.failed
    : displayStage === 'parsing'
      ? parsingLabel
      : displayStage === 'complete' && result?.chunks_created != null
        ? `${result.chunks_created} chunks indexed — ready to chat`
        : STAGE_LABELS[displayStage] || STAGE_LABELS.parsing;

  const docState = nodeState('document', displayStage, isFailed);
  const chunksState = nodeState('chunks', displayStage, isFailed);
  const embedState = nodeState('embeddings', displayStage, isFailed);
  const dbState = nodeState('vectordb', displayStage, isFailed);

  const connDocChunks = connectorState('document', 'chunks', displayStage, isFailed);
  const connChunksEmbed = connectorState('chunks', 'embeddings', displayStage, isFailed);
  const connEmbedDb = connectorState('embeddings', 'vectordb', displayStage, isFailed);

  const chunkYs = [52, 82, 112, 142, 172];

  return createPortal(
    <div className="rag-overlay" role="dialog" aria-modal="true" aria-label="Document processing pipeline">
      <div className="rag-overlay-backdrop" />
      <div className={`rag-overlay-panel ${isFailed ? 'rag-overlay-panel--failed' : ''}`}>
        <p className="rag-overlay-eyebrow">RAG Pipeline</p>
        <h2 className="rag-overlay-title">Processing your documents</h2>

        <div className="rag-flow-diagram">
          <svg viewBox="0 0 820 230" className="rag-flow-svg" aria-hidden="true">
            <defs>
              <filter id="rag-glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Document → Chunks connectors */}
            {chunkYs.map((y, i) => (
              <path
                key={`doc-chunk-${i}`}
                d={`M 168 115 Q 210 ${y + 8} 248 ${y + 12}`}
                className={`rag-connector rag-connector--${connDocChunks}`}
                fill="none"
              />
            ))}

            {/* Chunks → Embeddings */}
            {chunkYs.map((y, i) => (
              <path
                key={`chunk-embed-${i}`}
                d={`M 332 ${y + 12} Q 390 115 448 115`}
                className={`rag-connector rag-connector--${connChunksEmbed}`}
                fill="none"
              />
            ))}

            {/* Embeddings → Vector DB */}
            <path
              d="M 532 115 L 598 115"
              className={`rag-connector rag-connector--${connEmbedDb}`}
              fill="none"
            />

            {/* Document node */}
            <g className={`rag-node rag-node--${docState}`} filter={docState === 'active' ? 'url(#rag-glow)' : undefined}>
              <rect x="24" y="78" width="144" height="74" rx="12" className="rag-node-shape" />
              <text x="96" y="102" className="rag-node-label">User Input</text>
              <text x="96" y="122" className="rag-node-sublabel">
                {truncateName(primaryFile)}
              </text>
              {docState === 'done' ? (
                <text x="96" y="140" className="rag-node-check">✓</text>
              ) : fileNames.length > 1 ? (
                <text x="96" y="140" className="rag-node-meta">
                  +{fileNames.length - 1} more
                </text>
              ) : null}
            </g>

            {/* Chunk nodes */}
            {chunkYs.map((y, i) => {
              const chunkActive = chunksState === 'active';
              const lit = chunksState === 'done' || (chunkActive && i < chunkLitCount);
              return (
                <g
                  key={`chunk-${i}`}
                  className={`rag-node rag-node--chunk rag-node--${chunksState} ${lit ? 'rag-node--lit' : ''}`}
                  style={chunkActive ? { animationDelay: `${i * 0.15}s` } : undefined}
                >
                  <rect x="252" y={y} width="80" height="24" rx="8" className="rag-node-shape" />
                  <text x="292" y={y + 16} className="rag-node-label rag-node-label--sm">
                    chunk {i + 1}
                  </text>
                </g>
              );
            })}

            {/* Embeddings diamond */}
            <g
              className={`rag-node rag-node--${embedState}`}
              filter={embedState === 'active' ? 'url(#rag-glow)' : undefined}
            >
              <polygon points="490,78 532,115 490,152 448,115" className="rag-node-shape" />
              <text x="490" y="112" className="rag-node-label">Embed-</text>
              <text x="490" y="126" className="rag-node-label">dings</text>
              {embedState === 'done' && (
                <text x="490" y="140" className="rag-node-check">✓</text>
              )}
            </g>

            {/* Vector DB cylinder */}
            <g
              className={`rag-node rag-node--${dbState}`}
              filter={dbState === 'active' ? 'url(#rag-glow)' : undefined}
            >
              <ellipse cx="668" cy="88" rx="52" ry="14" className="rag-node-shape rag-cylinder-top" />
              <rect x="616" y="88" width="104" height="88" className="rag-node-shape rag-cylinder-body" />
              <ellipse cx="668" cy="176" rx="52" ry="14" className="rag-node-shape rag-cylinder-bottom" />
              <text x="668" y="72" className="rag-node-label">Vector DB</text>
              {[
                [640, 108], [672, 118], [652, 138], [688, 128], [630, 158],
                [670, 152], [700, 108], [645, 168], [685, 162], [658, 122],
              ].map(([cx, cy], i) => (
                <rect
                  key={`vec-${i}`}
                  x={cx}
                  y={cy}
                  width="10"
                  height="6"
                  rx="1"
                  className={`rag-vector-dot ${dbState !== 'pending' ? 'rag-vector-dot--visible' : ''}`}
                  style={{ animationDelay: `${i * 0.08}s` }}
                />
              ))}
              {dbState === 'done' && (
                <text x="668" y="200" className="rag-node-check">✓</text>
              )}
            </g>
          </svg>
        </div>

        <p className={`rag-overlay-status ${isFailed ? 'rag-overlay-status--error' : ''}`}>
          {statusLabel}
        </p>

        {isFailed && onClose && (
          <button type="button" className="btn primary rag-overlay-close" onClick={onClose}>
            Close
          </button>
        )}
      </div>
    </div>,
    document.body,
  );
}
