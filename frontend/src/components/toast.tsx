import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { CheckCircle2, XCircle } from "lucide-react";

type Toast = { id: number; message: string; kind: "success" | "error" };
const ToastContext = createContext<{ showToast: (message: string, kind?: Toast["kind"]) => void } | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const showToast = useCallback((message: string, kind: Toast["kind"] = "success") => {
    const id = Date.now(); setToasts((values) => [...values, { id, message, kind }]);
    window.setTimeout(() => setToasts((values) => values.filter((item) => item.id !== id)), 4500);
  }, []);
  return <ToastContext.Provider value={{ showToast }}>{children}<div className="fixed right-4 top-4 z-50 space-y-2" aria-live="polite">{toasts.map((toast) => <div key={toast.id} className={`flex max-w-sm items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium text-white shadow-lg ${toast.kind === "success" ? "bg-teal-700" : "bg-rose-700"}`}>{toast.kind === "success" ? <CheckCircle2 size={18} /> : <XCircle size={18} />}<span>{toast.message}</span></div>)}</div></ToastContext.Provider>;
}

export function useToast() { const value = useContext(ToastContext); if (!value) throw new Error("useToast must be used within ToastProvider"); return value; }
