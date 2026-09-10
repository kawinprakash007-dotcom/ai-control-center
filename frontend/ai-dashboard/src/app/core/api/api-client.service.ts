import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError, timeout, catchError } from 'rxjs';

export type AtlasApiErrorKind = 'timeout' | 'network' | 'http' | 'policy_denial' | 'unknown';

export class AtlasApiError extends Error {
  kind: AtlasApiErrorKind;
  status?: number;
  detail?: any;

  constructor(
    message: string,
    kind: AtlasApiErrorKind = 'unknown',
    status?: number,
    detail?: any
  ) {
    super(message);
    this.name = 'AtlasApiError';
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }
}

@Injectable({
  providedIn: 'root'
})
export class ApiClientService {
  private defaultBaseUrl = 'http://127.0.0.1:8000';
  private defaultToken = 'atlas_dev_secret_token';
  private defaultRequestTimeoutMs = 15000;
  private defaultChatTimeoutMs = 60000; // 60 seconds aligned with CognitiveRuntime turn limits

  constructor(private http: HttpClient) {}

  getBaseUrl(): string {
    if (typeof window !== 'undefined' && window.localStorage) {
      return localStorage.getItem('atlas_api_base_url') || this.defaultBaseUrl;
    }
    return this.defaultBaseUrl;
  }

  setBaseUrl(url: string): void {
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem('atlas_api_base_url', url.trim().replace(/\/+$/, ''));
    }
  }

  getAuthToken(): string {
    if (typeof window !== 'undefined' && window.localStorage) {
      return localStorage.getItem('atlas_auth_token') || this.defaultToken;
    }
    return this.defaultToken;
  }

  setAuthToken(token: string): void {
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem('atlas_auth_token', token.trim());
    }
  }

  getChatTimeoutMs(): number {
    if (typeof window !== 'undefined' && window.localStorage) {
      const stored = localStorage.getItem('atlas_chat_timeout_ms');
      if (stored) {
        const val = parseInt(stored, 10);
        if (!isNaN(val) && val > 0) return val;
      }
    }
    return this.defaultChatTimeoutMs;
  }

  setChatTimeoutMs(ms: number): void {
    if (typeof window !== 'undefined' && window.localStorage && ms > 0) {
      localStorage.setItem('atlas_chat_timeout_ms', String(ms));
    }
  }

  private getHeaders(includeAuth: boolean = true): HttpHeaders {
    let headers = new HttpHeaders({
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    });
    if (includeAuth) {
      headers = headers.set('Authorization', `Bearer ${this.getAuthToken()}`);
    }
    return headers;
  }

  get<T>(path: string, includeAuth: boolean = true, customTimeoutMs?: number): Observable<T> {
    const url = `${this.getBaseUrl()}${path.startsWith('/') ? path : '/' + path}`;
    const effectiveTimeout = customTimeoutMs && customTimeoutMs > 0 ? customTimeoutMs : this.defaultRequestTimeoutMs;
    return this.http.get<T>(url, { headers: this.getHeaders(includeAuth) }).pipe(
      timeout(effectiveTimeout),
      catchError((err) => this.handleError(err))
    );
  }

  post<T>(path: string, body: any, includeAuth: boolean = true, customTimeoutMs?: number): Observable<T> {
    const url = `${this.getBaseUrl()}${path.startsWith('/') ? path : '/' + path}`;
    const effectiveTimeout = customTimeoutMs && customTimeoutMs > 0 ? customTimeoutMs : this.defaultRequestTimeoutMs;
    return this.http.post<T>(url, body, { headers: this.getHeaders(includeAuth) }).pipe(
      timeout(effectiveTimeout),
      catchError((err) => this.handleError(err))
    );
  }

  private handleError(error: HttpErrorResponse | any): Observable<never> {
    // 1. Timeout detection (RxJS TimeoutError)
    const isTimeout =
      (error && error.name === 'TimeoutError') ||
      (error && typeof error.message === 'string' && error.message.toLowerCase().includes('timeout'));

    if (isTimeout) {
      return throwError(
        () =>
          new AtlasApiError(
            'ATLAS is still processing the cognitive turn (request timed out on client).',
            'timeout'
          )
      );
    }

    // 2. HTTP Error Response
    if (error instanceof HttpErrorResponse) {
      if (error.status === 0) {
        return throwError(
          () =>
            new AtlasApiError(
              'Backend unreachable or connection refused. Please verify ATLAS server is running.',
              'network',
              0
            )
        );
      }

      const detail =
        error.error && typeof error.error.detail === 'string'
          ? error.error.detail
          : error.message || `HTTP ${error.status}`;

      if (
        error.status === 403 ||
        (typeof detail === 'string' && detail.toLowerCase().includes('policy'))
      ) {
        return throwError(
          () => new AtlasApiError(detail, 'policy_denial', error.status, error.error)
        );
      }

      return throwError(
        () => new AtlasApiError(detail, 'http', error.status, error.error)
      );
    }

    // 3. Fallback generic error
    const msg = error && error.message ? error.message : 'An unknown network error occurred';
    return throwError(() => new AtlasApiError(msg, 'unknown'));
  }
}
