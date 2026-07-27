import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ragApi } from "../services/api";

export const queryKeys = { health: ["health"] as const, models: ["models"] as const, config: ["config"] as const, documents: ["documents"] as const };
export const useHealth = () => useQuery({ queryKey: queryKeys.health, queryFn: ragApi.health, refetchInterval: 30_000 });
export const useModels = () => useQuery({ queryKey: queryKeys.models, queryFn: ragApi.models });
export const useConfig = () => useQuery({ queryKey: queryKeys.config, queryFn: ragApi.config });
export const useDocuments = () => useQuery({ queryKey: queryKeys.documents, queryFn: ragApi.documents });
export function useDocumentMutations() { const client = useQueryClient(); return { upload: useMutation({ mutationFn: ({ files, onProgress }: { files: File[]; onProgress: (value: number) => void }) => ragApi.uploadDocuments(files, onProgress), onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.documents }) }), remove: useMutation({ mutationFn: ragApi.deleteDocument, onSuccess: () => { client.invalidateQueries({ queryKey: queryKeys.documents }); client.invalidateQueries({ queryKey: queryKeys.health }); } }) }; }
