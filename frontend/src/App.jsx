import { useEffect, useRef } from "react";
import Icon from "./components/Icon.jsx";
import PreviewScreen from "./components/PreviewScreen.jsx";
import ResultScreen from "./components/ResultScreen.jsx";
import ScanScreen from "./components/ScanScreen.jsx";
import StatusBanner from "./components/StatusBanner.jsx";
import { useBackendStatus } from "./hooks/useBackendStatus.js";
import { useScan } from "./hooks/useScan.js";

export default function App() {
  const backend = useBackendStatus();
  const { state, pick, analyze, reset, backToPreview } = useScan();
  const { step, previewUrl, result, error } = state;

  // Move focus to the new screen's heading so screen readers announce the change.
  const mainRef = useRef(null);
  const firstRender = useRef(true);
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    mainRef.current?.querySelector("h1")?.focus();
  }, [step]);

  return (
    <div className="shell">
      <header className="masthead">
        <Icon name="leaf" size={26} />
        <span>Krishi Mithra</span>
      </header>

      <StatusBanner status={backend.status} onRetry={backend.recheck} />

      <main ref={mainRef}>
        {step === "scan" && <ScanScreen onFile={pick} error={error} />}
        {(step === "preview" || step === "analyzing") && (
          <PreviewScreen
            previewUrl={previewUrl}
            analyzing={step === "analyzing"}
            error={error}
            onChooseAnother={reset}
            onAnalyze={analyze}
          />
        )}
        {step === "result" && result && <ResultScreen previewUrl={previewUrl} result={result} onBack={backToPreview} onReset={reset} />}
      </main>
    </div>
  );
}
