import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { of, firstValueFrom } from 'rxjs';
import { AtlasApiService } from './atlas-api.service';
import { ApiClientService } from './api-client.service';

describe('AtlasApiService', () => {
  let service: AtlasApiService;
  let apiClientSpy: { get: ReturnType<typeof vi.fn>; post: ReturnType<typeof vi.fn> };

  beforeEach(() => {
    apiClientSpy = {
      get: vi.fn(),
      post: vi.fn()
    };

    TestBed.configureTestingModule({
      providers: [
        AtlasApiService,
        provideHttpClient(),
        { provide: ApiClientService, useValue: apiClientSpy }
      ]
    });

    service = TestBed.inject(AtlasApiService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should call getHealth without auth requirement', async () => {
    const mockHealth = { status: 'healthy', version: '1.0.0', uptime_seconds: 120, components: {} } as any;
    apiClientSpy.get.mockReturnValue(of(mockHealth));

    const res = await firstValueFrom(service.getHealth());
    expect(res).toEqual(mockHealth);
    expect(apiClientSpy.get).toHaveBeenCalledWith('/api/v1/health', false);
  });

  it('should call getReady probe without auth requirement', async () => {
    const mockReady = { status: 'ready', initialized: true, subsystems: {} } as any;
    apiClientSpy.get.mockReturnValue(of(mockReady));

    const res = await firstValueFrom(service.getReady());
    expect(res).toEqual(mockReady);
    expect(apiClientSpy.get).toHaveBeenCalledWith('/api/v1/ready', false);
  });

  it('should dispatch device command to correct URL', async () => {
    const mockResult = { command_id: 'cmd-1', status: 'success', success: true, message: 'Executed' };
    apiClientSpy.post.mockReturnValue(of(mockResult));

    const req = { command: 'takeoff', parameters: { altitude: 10 } };
    const res = await firstValueFrom(service.sendCommand('ATLAS_DRONE_01', req));
    expect(res).toEqual(mockResult);
    expect(apiClientSpy.post).toHaveBeenCalledWith('/api/v1/devices/ATLAS_DRONE_01/command', req);
  });

  it('should query world state', async () => {
    const mockWs = { state_id: 'ws-1', version: 1, timestamp: 100, entities: [], conditions: [], relationships: [], metadata: {} } as any;
    apiClientSpy.get.mockReturnValue(of(mockWs));

    const res = await firstValueFrom(service.getWorldState());
    expect(res).toEqual(mockWs);
    expect(apiClientSpy.get).toHaveBeenCalledWith('/api/v1/world/state');
  });
});
