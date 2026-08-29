import createClient from "openapi-fetch";
import type { paths } from "../types/api";

// Dev: proxied through Vite (vite.config.ts) so "" resolves same-origin.
// Prod: backend URL is baked in at build time (ARCHITECTURE.md §11).
const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

export const apiClient = createClient<paths>({ baseUrl });
