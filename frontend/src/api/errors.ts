import type { components } from "../types/api";

type ErrorResponse = components["schemas"]["ErrorResponse"];
type HTTPValidationError = components["schemas"]["HTTPValidationError"];

export type ApiErrorBody = ErrorResponse | HTTPValidationError;

/**
 * The backend's single error shape (ARCHITECTURE.md §6, backend.md) covers
 * every AppError, but FastAPI's own 422 request-validation errors arrive as
 * HTTPValidationError instead (see app/main.py's validation_error_handler).
 * Normalise both into one readable message for the UI.
 */
export function describeApiError(body: ApiErrorBody | undefined): string {
  if (!body) return "Request failed.";
  if ("message" in body && body.message) return body.message;
  if ("detail" in body && body.detail) {
    const combined = body.detail.map((entry) => entry.msg).join("; ");
    if (combined) return combined;
  }
  return "Request failed.";
}
