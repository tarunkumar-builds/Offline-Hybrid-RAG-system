import { Navigate, Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/app-layout";
import { ChatPage } from "./pages/chat-page";
import { DashboardPage } from "./pages/dashboard-page";
import { DocumentsPage } from "./pages/documents-page";
import { EvaluationPage } from "./pages/evaluation-page";
import { SystemPage } from "./pages/system-page";

export default function App() { return <AppLayout><Routes><Route path="/" element={<DashboardPage />} /><Route path="/documents" element={<DocumentsPage />} /><Route path="/chat" element={<ChatPage />} /><Route path="/evaluation" element={<EvaluationPage />} /><Route path="/system" element={<SystemPage />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes></AppLayout>; }
