import { LEVELS, adviceIntro, headline, possibilities, rankWord } from "../lib/copy.js";
import Icon from "./Icon.jsx";

export default function ResultScreen({ previewUrl, result, onBack, onReset }) {
  const level = result.confidence_level;
  const list = possibilities(result);

  return (
    <section className="screen" aria-labelledby="result-title">
      <div className="screen-head">
        <button type="button" className="icon-button" onClick={onBack} aria-label="Back to the photo">
          <Icon name="back" />
        </button>
        <p className="screen-head-title">Result</p>
      </div>

      <figure className="specimen">
        <div className="photo-frame photo-frame-short">
          <img src={previewUrl} alt="The leaf you checked" />
        </div>
        <figcaption className={`tag tag-${level}`}>
          <span className="tag-hole" aria-hidden="true" />
          <h1 id="result-title" tabIndex={-1}>{headline(result)}</h1>
          <p className="tag-level">{LEVELS[level].label}</p>
          <p className="tag-message">{result.message}</p>
        </figcaption>
      </figure>

      {result.needs_expert && (
        <div className="expert">
          <Icon name="people" size={26} />
          <p>Show this plant to your nearest KVK or Krishi Bhavan officer for a reliable diagnosis.</p>
        </div>
      )}

      {result.advice?.length > 0 && (
        <section className="block">
          <h2>{adviceIntro(result)}</h2>
          <ul className="steps">
            {result.advice.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
          {!result.is_healthy && <p className="footnote">Ask KVK or Krishi Bhavan before using any chemical treatment.</p>}
        </section>
      )}

      {list.items.length > 0 && (
        <section className="block">
          <h2>{list.title}</h2>
          <ol className="ranked">
            {list.items.map((p) => (
              <li key={p.label}>
                <span className={`chip chip-${Math.min(p.rank, 3)}`} aria-hidden="true" />
                <span className="ranked-name">{p.name}</span>
                <span className="ranked-word">{rankWord(p.rank)}</span>
              </li>
            ))}
          </ol>
          {level !== "high" && <p className="footnote">These are only guesses. A clearer photo or an expert can confirm.</p>}
        </section>
      )}

      <div className="actions">
        <button type="button" className="btn btn-primary" onClick={onReset}>
          <Icon name="camera" /> Check another leaf
        </button>
      </div>
    </section>
  );
}
