/**
 * Centralised API base-URL configuration.
 *
 * In Docker / production:  set VITE_API_URL in the frontend service environment.
 * In local dev:            leave unset — falls back to http://127.0.0.1:8000
 *
 * Usage:  import { API } from '../config/api';
 */
export const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
