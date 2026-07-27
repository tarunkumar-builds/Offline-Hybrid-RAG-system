import { useState } from "react";
import { ChevronDown, ChevronUp, FileText } from "lucide-react";
import type { RetrievedChunk } from "../types/api";

export function RetrievedChunks({ chunks }: { chunks: RetrievedChunk[] }) {
  const [expanded, setExpanded] = useState(false);
  if (!chunks.length) return null;
  return <section className="card"><button className="flex w-full items-center justify-between text-left" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded}><span className="font-semibold">Retrieved chunks <span className="ml-1 text-sm font-normal text-slate-500">({chunks.length})</span></span>{expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}</button>{expanded && <div className="mt-4 space-y-3">{chunks.map((chunk) => <article key={chunk.chunk_id} className="rounded-xl border border-slate-200 p-4 dark:border-slate-700"><div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500"><span className="inline-flex items-center gap-1"><FileText size={13} />{chunk.document_name}</span><span>Page {chunk.page_number}</span><span>Chunk {chunk.chunk_number}</span><span>Rerank {chunk.rerank_score.toFixed(3)}</span><span>Retrieval {chunk.retrieval_score.toFixed(3)}</span></div><p className="whitespace-pre-wrap text-sm leading-6 text-slate-700 dark:text-slate-200">{chunk.text}</p></article>)}</div>}</section>;
}
