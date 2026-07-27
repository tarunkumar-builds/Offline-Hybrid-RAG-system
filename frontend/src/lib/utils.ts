import axios from "axios";
import type { ApiError } from "../types/api";

export const formatDate = (value: string) => new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
export const formatSeconds = (value: number) => `${value < 1 ? (value * 1000).toFixed(0) + " ms" : value.toFixed(2) + " s"}`;
export const apiErrorMessage = (error: unknown) => {
  if (axios.isAxiosError<ApiError>(error)) {
    const detail = error.response?.data?.detail;
    return typeof detail === "string" ? detail : detail?.[0]?.msg ?? "The request could not be completed.";
  }
  return "The request could not be completed.";
};
