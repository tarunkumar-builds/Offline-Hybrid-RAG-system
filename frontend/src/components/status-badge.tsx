import { Circle } from "lucide-react";

export function StatusBadge({ value }: { value: string }) {
  const healthy = value === "healthy" || value === "ready";
  return <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${healthy ? "bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-300" : "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"}`}><Circle size={8} fill="currentColor" />{value}</span>;
}
