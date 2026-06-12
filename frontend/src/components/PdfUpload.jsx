import { useRef, useState } from 'react';
import { pollUploadUntilComplete, uploadPdfs } from '../api/client';
import RagFlowOverlay from './RagFlowOverlay';

function filterPdfFiles(fileList) {
  return Array.from(fileList || []).filter(
    (file) => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf'),
  );
}

export default function PdfUpload({ onUploadComplete, disabled, variant = 'sidebar', autoUpload = false }) {
  const inputRef = useRef(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);
  const [lastResult, setLastResult] = useState(null);
  const [overlayOpen, setOverlayOpen] = useState(false);
  const [pipelineStage, setPipelineStage] = useState('uploading');
  const [overlayFiles, setOverlayFiles] = useState([]);
  const [filesDone, setFilesDone] = useState(0);
  const [fileCount, setFileCount] = useState(0);
  const [overlayResult, setOverlayResult] = useState(null);
  const [overlayError, setOverlayError] = useState(null);

  const isHero = variant === 'hero';

  const resetOverlay = () => {
    setOverlayOpen(false);
    setPipelineStage('uploading');
    setOverlayFiles([]);
    setFilesDone(0);
    setFileCount(0);
    setOverlayResult(null);
    setOverlayError(null);
  };

  const setFiles = (files) => {
    setSelectedFiles(files);
    setError(null);
    setLastResult(null);
  };

  const handleFileChange = (event) => {
    if (disabled || uploading) return;

    const files = filterPdfFiles(event.target.files);
    if (!files.length && event.target.files?.length) {
      setError('Please select PDF files only.');
      return;
    }
    setFiles(files);
    if ((autoUpload || isHero) && files.length) {
      handleUpload(files);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    if (!disabled && !uploading) setDragging(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    if (disabled || uploading) return;

    const files = filterPdfFiles(event.dataTransfer.files);
    if (!files.length) {
      setError('Please drop PDF files only.');
      return;
    }

    setFiles(files);
    if (autoUpload || isHero) {
      handleUpload(files);
    }
  };

  const handleUpload = async (filesOverride) => {
    const files = filesOverride || selectedFiles;
    if (!files.length) return;

    setUploading(true);
    setError(null);
    setOverlayOpen(true);
    setPipelineStage('uploading');
    setOverlayFiles(files);
    setFilesDone(0);
    setFileCount(files.length);
    setOverlayResult(null);
    setOverlayError(null);

    try {
      const accepted = await uploadPdfs(files);
      setFileCount(accepted.file_count || files.length);
      setPipelineStage('parsing');

      const result = await pollUploadUntilComplete(accepted.task_id, {
        onStatus: (status) => {
          if (status.stage) {
            setPipelineStage(status.stage === 'queued' ? 'parsing' : status.stage);
          }
          if (status.files_done != null) setFilesDone(status.files_done);
          if (status.file_count != null) setFileCount(status.file_count);
        },
      });

      setLastResult(result);
      setSelectedFiles([]);
      if (inputRef.current) inputRef.current.value = '';

      if (result.status === 'failed') {
        throw new Error(result.error || 'Processing failed');
      }

      setPipelineStage('complete');
      setOverlayResult(result);
    } catch (err) {
      setError(err.message);
      setOverlayError(err.message);
      setPipelineStage('failed');
    } finally {
      setUploading(false);
    }
  };

  const dropLabel = selectedFiles.length
    ? `${selectedFiles.length} file(s) selected`
    : isHero
      ? 'Drop PDFs here or click to browse'
      : 'Choose PDF files';

  return (
    <>
      <div className={`upload-panel ${isHero ? 'upload-panel-hero' : ''}`}>
        {!isHero && (
          <>
            <h2>Upload PDFs</h2>
            <p className="panel-desc">Add documents to build your knowledge base (max 50MB per file).</p>
          </>
        )}

        <label
          className={`file-drop ${isHero ? 'file-drop-hero' : ''} ${dragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf"
            multiple
            onChange={handleFileChange}
            disabled={disabled || uploading}
          />
          <span className="file-drop-label">{dropLabel}</span>
          {isHero && <span className="file-drop-hint">PDF only · up to 50MB per file</span>}
        </label>

        {selectedFiles.length > 0 && !isHero && (
          <ul className="file-list">
            {selectedFiles.map((file) => (
              <li key={file.name}>{file.name}</li>
            ))}
          </ul>
        )}

        {!isHero && (
          <button
            type="button"
            className="btn primary"
            onClick={() => handleUpload()}
            disabled={disabled || uploading || !selectedFiles.length}
          >
            {uploading ? 'Processing…' : 'Process Documents'}
          </button>
        )}

        {isHero && selectedFiles.length > 0 && !uploading && (
          <button
            type="button"
            className="btn primary hero-upload-btn"
            onClick={() => handleUpload()}
            disabled={disabled || uploading}
          >
            Process {selectedFiles.length} file(s)
          </button>
        )}

        {error && !overlayOpen && <p className="message error">{error}</p>}
        {lastResult?.status === 'completed' && !isHero && !overlayOpen && (
          <p className="message success">
            Processed {lastResult.file_count} file(s), {lastResult.chunks_created} chunks created.
          </p>
        )}
      </div>

      <RagFlowOverlay
        open={overlayOpen}
        stage={pipelineStage}
        files={overlayFiles}
        filesDone={filesDone}
        fileCount={fileCount}
        result={overlayResult}
        error={overlayError}
        onClose={resetOverlay}
        onDismiss={() => {
          const completed = overlayResult;
          resetOverlay();
          if (completed) onUploadComplete?.(completed);
        }}
      />
    </>
  );
}
