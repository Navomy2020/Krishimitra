const COPY = {
  offline: "Can't reach the Krishi Mithra server. Start the backend, then try again.",
  degraded: "The server is running but the disease model did not load. Check the backend window for the reason.",
};

export default function StatusBanner({ status, onRetry }) {
  if (!COPY[status]) return null;
  return (
    <div className="banner" role="alert">
      <p>{COPY[status]}</p>
      <button type="button" className="link-button" onClick={onRetry}>
        Check again
      </button>
    </div>
  );
}
