import { useEffect, useState } from "react";
import { fetchHealth } from "../api";

export function ApiStatus() {
  const [status, setStatus] = useState<"checking" | "ok" | "error">("checking");

  useEffect(() => {
    fetchHealth()
      .then(() => setStatus("ok"))
      .catch(() => setStatus("error"));
  }, []);

  if (status === "checking") return null;
  return (
    <span
      className={`text-xs px-2 py-0.5 rounded-full ${
        status === "ok" ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
      }`}
      title={status === "ok" ? "API connected" : "API unavailable"}
    >
      {status === "ok" ? "API" : "Offline"}
    </span>
  );
}
