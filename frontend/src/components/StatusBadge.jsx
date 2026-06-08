export default function StatusBadge({ status }) {
  const ready = status?.vectorstore_loaded;

  return (
    <div className={`status-badge ${ready ? 'ready' : 'pending'}`}>
      <span className="status-dot" />
      {ready ? 'Ready to chat' : 'Upload documents first'}
    </div>
  );
}
