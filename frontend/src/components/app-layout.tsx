import { useState, type ReactNode } from "react";
import { Bot, FileText, Gauge, Menu, MessageSquareText, Moon, ServerCog, Sun, X } from "lucide-react";
import { NavLink } from "react-router-dom";
import { useTheme } from "./theme-provider";

const links = [
  { to: "/", label: "Dashboard", icon: Gauge }, { to: "/documents", label: "Documents", icon: FileText },
  { to: "/chat", label: "Ask questions", icon: MessageSquareText }, { to: "/evaluation", label: "Evaluation", icon: Bot }, { to: "/system", label: "System status", icon: ServerCog },
];

export function AppLayout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false); const { theme, toggleTheme } = useTheme();
  const navigation = <nav className="space-y-1">{links.map(({ to, label, icon: Icon }) => <NavLink onClick={() => setOpen(false)} key={to} to={to} end={to === "/"} className={({ isActive }) => `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition ${isActive ? "bg-teal-700 text-white" : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"}`}><Icon size={18} />{label}</NavLink>)}</nav>;
  return <div className="min-h-screen"><aside className="fixed inset-y-0 left-0 z-40 hidden w-64 border-r border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900 md:block"><Brand />{navigation}<div className="absolute bottom-6 text-xs text-slate-400">Offline Hybrid RAG · v1</div></aside>{open && <div className="fixed inset-0 z-50 bg-slate-950/40 md:hidden" onClick={() => setOpen(false)}><aside className="h-full w-72 bg-white p-5 dark:bg-slate-900" onClick={(event) => event.stopPropagation()}><div className="flex justify-end"><button className="btn-secondary p-2" onClick={() => setOpen(false)} aria-label="Close navigation"><X size={18} /></button></div><Brand />{navigation}</aside></div>}<main className="min-h-screen md:pl-64"><header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-slate-200 bg-slate-50/90 px-4 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90 md:px-8"><button className="btn-secondary p-2 md:hidden" onClick={() => setOpen(true)} aria-label="Open navigation"><Menu size={18} /></button><p className="hidden text-sm text-slate-500 sm:block">Private, local retrieval and grounded generation</p><button className="btn-secondary p-2" onClick={toggleTheme} aria-label="Toggle color theme">{theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}</button></header><div className="mx-auto max-w-7xl p-4 md:p-8">{children}</div><footer className="border-t border-slate-200 px-8 py-5 text-center text-xs text-slate-500 dark:border-slate-800">Offline Hybrid RAG · your documents stay local</footer></main></div>;
}

function Brand() { return <div className="mb-8 flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-teal-700 text-white"><Bot size={21} /></span><div><h1 className="font-bold tracking-tight">Hybrid RAG</h1><p className="text-xs text-slate-500">Offline workspace</p></div></div>; }
