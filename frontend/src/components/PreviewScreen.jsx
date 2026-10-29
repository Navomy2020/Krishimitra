import Icon from "./Icon.jsx";

export default function PreviewScreen({ previewUrl, analyzing, error, onChooseAnother, onAnalyze }) {
  return (
    <section className="screen" aria-labelledby="preview-title">
      <div className="screen-head">
        <button type="button" className="icon-button" onClick={onChooseAnother} aria-label={analyzing ? "Cancel and choose another photo" : "Choose another photo"}>
          <Icon name="back" />
        </button>
        <h1 id="preview-title" tabIndex={-1}>Is the leaf sharp and fully in view?</h1>
      </div>

      <div className={`photo-frame${analyzing ? " is-scanning" : ""}`}>
        <img src={previewUrl} alt="The leaf photo you chose" />
        {analyzing && <div className="sweep" aria-hidden="true" />}
      </div>

      <p className="status" role="status" aria-live="polite">
        {analyzing ? "Checking the leaf..." : ""}
      </p>

      {error && (
        <p className="notice" role="alert">
          {error}
        </p>
      )}

      <div className="actions actions-row">
        <button type="button" className="btn btn-quiet" onClick={onChooseAnother}>
          {analyzing ? "Cancel" : "Choose another"}
        </button>
        <button type="button" className="btn btn-primary" onClick={onAnalyze} disabled={analyzing}>
          {analyzing ? "Checking..." : error ? <><Icon name="retry" /> Try again</> : "Check this leaf"}
        </button>
      </div>
    </section>
  );
}
