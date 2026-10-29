// The whole flow in one reducer: scan -> preview -> analyzing -> result.

export const initialState = { step: "scan", file: null, previewUrl: null, result: null, error: null };

export function scanReducer(state, action) {
  switch (action.type) {
    case "PICK": // a new photo always starts a fresh check
      return { ...initialState, step: "preview", file: action.file, previewUrl: action.previewUrl };
    case "PICK_ERROR":
      return { ...state, error: action.message };
    case "ANALYZE_START":
      return { ...state, step: "analyzing", error: null };
    case "ANALYZE_OK":
      return { ...state, step: "result", result: action.result, error: null };
    case "ANALYZE_FAIL":
      return { ...state, step: "preview", error: action.message };
    case "BACK_TO_PREVIEW":
      return { ...state, step: "preview", error: null };
    case "RESET":
      return initialState;
    default:
      return state;
  }
}
