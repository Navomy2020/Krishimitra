import { useCallback, useEffect, useReducer, useRef } from "react";
import { api as defaultApi } from "../api.js";
import { isImageFile, prepareImage } from "../lib/image.js";
import { initialState, scanReducer } from "../lib/scanState.js";

export function useScan(api = defaultApi) {
  const [state, dispatch] = useReducer(scanReducer, initialState);
  const abortRef = useRef(null);

  // Free the preview image's memory when it is replaced or the page closes.
  useEffect(() => () => state.previewUrl && URL.revokeObjectURL(state.previewUrl), [state.previewUrl]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const pick = useCallback(
    (file) => {
      if (!isImageFile(file)) {
        dispatch({ type: "PICK_ERROR", message: "That file is not a photo. Choose a JPG or PNG image of a leaf." });
        return;
      }
      cancel();
      dispatch({ type: "PICK", file, previewUrl: URL.createObjectURL(file) });
    },
    [cancel],
  );

  const analyze = useCallback(async () => {
    if (!state.file || state.step === "analyzing") return;
    cancel();
    const controller = new AbortController();
    abortRef.current = controller;
    dispatch({ type: "ANALYZE_START" });
    try {
      const photo = await prepareImage(state.file);
      const result = await api.predict(photo, { signal: controller.signal });
      if (controller.signal.aborted) return;
      dispatch({ type: "ANALYZE_OK", result });
    } catch (err) {
      if (controller.signal.aborted) return; // the user went back; say nothing
      dispatch({ type: "ANALYZE_FAIL", message: err?.message || "Something went wrong. Try again." });
    }
  }, [api, cancel, state.file, state.step]);

  const reset = useCallback(() => {
    cancel();
    dispatch({ type: "RESET" });
  }, [cancel]);

  const backToPreview = useCallback(() => {
    cancel();
    dispatch({ type: "BACK_TO_PREVIEW" });
  }, [cancel]);

  useEffect(() => cancel, [cancel]);

  return { state, pick, analyze, reset, backToPreview };
}
