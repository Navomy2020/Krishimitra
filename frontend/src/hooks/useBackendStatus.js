import { useCallback, useEffect, useState } from "react";
import { api as defaultApi } from "../api.js";

// "checking" | "ready" | "degraded" (server up, model not loaded) | "offline"
export function useBackendStatus(api = defaultApi) {
  const [status, setStatus] = useState("checking");

  const check = useCallback(async () => {
    setStatus("checking");
    try {
      const health = await api.health();
      setStatus(health.model_loaded ? "ready" : "degraded");
    } catch {
      setStatus("offline");
    }
  }, [api]);

  useEffect(() => {
    check();
  }, [check]);

  return { status, recheck: check };
}
