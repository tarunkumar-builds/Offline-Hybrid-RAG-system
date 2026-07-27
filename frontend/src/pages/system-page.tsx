import { Server, Database, Cpu } from "lucide-react";
import type { ReactNode } from "react";
import { StatusBadge } from "../components/status-badge";
import { useConfig, useHealth, useModels } from "../hooks/use-rag";
import { Failure, Loading } from "./dashboard-page";

export function SystemPage() {
  const health = useHealth(); const models = useModels(); const config = useConfig();
  if (health.isLoading || models.isLoading) return <Loading label="Loading system status…" />;
  if (health.isError || models.isError || !health.data || !models.data) return <Failure onRetry={() => { health.refetch(); models.refetch(); config.refetch(); }} />;
  const data = health.data; const model = models.data;
  return <div className="space-y-6"><div><p className="text-sm font-semibold text-teal-700">System status</p><h2 className="mt-1 text-3xl font-bold">Local services at a glance</h2></div><div className="grid gap-4 md:grid-cols-3"><Tile icon={Server} label="Backend" value={<StatusBadge value={data.status} />} /><Tile icon={Cpu} label="Ollama" value={<StatusBadge value={data.ollama_status} />} /><Tile icon={Database} label="Index" value={`${data.indexed_documents} docs · ${data.vector_count} chunks`} /></div><section className="card"><h3 className="font-semibold">Configured models</h3><dl className="mt-4 divide-y divide-slate-100 dark:divide-slate-800"><Row label="Embedding model" value={model.embedding_model} /><Row label="Reranker" value={model.reranker_model} /><Row label="Local LLM" value={model.llm_model} /></dl></section><section className="card"><h3 className="font-semibold">Public runtime configuration</h3><pre className="mt-4 overflow-x-auto rounded-xl bg-slate-950 p-4 text-xs leading-6 text-slate-100">{JSON.stringify(config.data?.configuration ?? {}, null, 2)}</pre></section></div>;
}
function Tile({ icon: Icon, label, value }: { icon: typeof Server; label: string; value: ReactNode }) { return <section className="card"><Icon className="text-teal-700" size={20} /><p className="mt-4 text-sm text-slate-500">{label}</p><div className="mt-2 font-semibold">{value}</div></section>; }
function Row({ label, value }: { label: string; value: string }) { return <div className="flex flex-col gap-1 py-3 sm:flex-row sm:justify-between"><dt className="text-sm text-slate-500">{label}</dt><dd className="break-all text-sm font-medium">{value}</dd></div>; }
