import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpErrorResponse } from '@angular/common/http';
import { ApiClientService, AtlasApiError } from './api-client.service';
import { firstValueFrom, throwError } from 'rxjs';

describe('ApiClientService Timeout & Error Handling', () => {
  let service: ApiClientService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [ApiClientService, provideHttpClient()]
    });
    service = TestBed.inject(ApiClientService);
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.removeItem('atlas_chat_timeout_ms');
    }
  });

  it('should default chat timeout to 60000ms', () => {
    expect(service.getChatTimeoutMs()).toBe(60000);
  });

  it('should allow configuring chat timeout via setChatTimeoutMs and localStorage', () => {
    service.setChatTimeoutMs(90000);
    expect(service.getChatTimeoutMs()).toBe(90000);
  });

  it('should classify timeout errors as kind: timeout', async () => {
    const timeoutErr = new Error('Timeout has occurred');
    timeoutErr.name = 'TimeoutError';

    // Test private handleError via any cast
    const obs = (service as any).handleError(timeoutErr);
    try {
      await firstValueFrom(obs);
      expect.unreachable('Should have thrown');
    } catch (e: any) {
      expect(e).toBeInstanceOf(AtlasApiError);
      expect(e.kind).toBe('timeout');
      expect(e.message).toContain('ATLAS is still processing');
    }
  });

  it('should classify connection refusal (status 0) as kind: network', async () => {
    const networkErr = new HttpErrorResponse({ status: 0, statusText: 'Unknown Error' });
    const obs = (service as any).handleError(networkErr);
    try {
      await firstValueFrom(obs);
      expect.unreachable('Should have thrown');
    } catch (e: any) {
      expect(e).toBeInstanceOf(AtlasApiError);
      expect(e.kind).toBe('network');
      expect(e.message).toContain('Backend unreachable');
    }
  });

  it('should classify policy denial as kind: policy_denial', async () => {
    const policyErr = new HttpErrorResponse({
      status: 403,
      error: { detail: 'Policy denied execution: Action forbidden' }
    });
    const obs = (service as any).handleError(policyErr);
    try {
      await firstValueFrom(obs);
      expect.unreachable('Should have thrown');
    } catch (e: any) {
      expect(e).toBeInstanceOf(AtlasApiError);
      expect(e.kind).toBe('policy_denial');
      expect(e.message).toContain('Policy denied');
    }
  });
});
