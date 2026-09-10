import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { of, throwError } from 'rxjs';
import { CommandBarComponent } from './command-bar.component';
import { AtlasApiService } from '../../../core/api/atlas-api.service';
import { AtlasApiError } from '../../../core/api/api-client.service';
import { AppStateService } from '../../../core/state/app-state.service';

describe('CommandBarComponent', () => {
  let apiSpy: any;
  let appStateMock: any;

  beforeEach(async () => {
    apiSpy = {
      postChat: vi.fn().mockReturnValue(of({
        turn_id: 'turn-test-1',
        status: 'success',
        response: 'Acknowledged operational directive.',
        execution_time: 150,
        trace_id: 'tr-1',
      })),
    };

    appStateMock = {
      isDemoMode: vi.fn().mockReturnValue(false),
    };

    await TestBed.configureTestingModule({
      imports: [CommandBarComponent],
      providers: [
        provideHttpClient(),
        { provide: AtlasApiService, useValue: apiSpy },
        { provide: AppStateService, useValue: appStateMock },
      ]
    }).compileComponents();
  });

  it('should render the command bar input', () => {
    const fixture = TestBed.createComponent(CommandBarComponent);
    fixture.detectChanges();
    const input = fixture.nativeElement.querySelector('.command-input') as HTMLInputElement;
    expect(input).toBeTruthy();
    expect(input.placeholder).toContain('Command drones');
  });

  it('should submit query to cognitive runtime via submitCommand()', () => {
    const fixture = TestBed.createComponent(CommandBarComponent);
    const component = fixture.componentInstance;
    component.userMessage = 'Scan perimeter';
    component.submitCommand();

    expect(apiSpy.postChat).toHaveBeenCalledWith('Scan perimeter');
    expect(component.submittedQuery).toBe('Scan perimeter');
    expect(component.lastResponse?.response).toBe('Acknowledged operational directive.');
  });

  it('should display processing message when timeout occurs', () => {
    apiSpy.postChat.mockReturnValue(
      throwError(() => new AtlasApiError('Timed out', 'timeout'))
    );

    const fixture = TestBed.createComponent(CommandBarComponent);
    const component = fixture.componentInstance;
    component.userMessage = 'Status report';
    component.submitCommand();

    expect(component.errorMessage).toContain('ATLAS is still processing');
    expect(component.lastResponse).toBeNull();
  });

  it('should display connection failure message when network is unreachable', () => {
    apiSpy.postChat.mockReturnValue(
      throwError(() => new AtlasApiError('Network down', 'network'))
    );

    const fixture = TestBed.createComponent(CommandBarComponent);
    const component = fixture.componentInstance;
    component.userMessage = 'Status report';
    component.submitCommand();

    expect(component.errorMessage).toContain('Connection failure');
  });

  it('should display policy restriction message when action is denied', () => {
    apiSpy.postChat.mockReturnValue(
      throwError(() => new AtlasApiError('Capability open_app forbidden', 'policy_denial'))
    );

    const fixture = TestBed.createComponent(CommandBarComponent);
    const component = fixture.componentInstance;
    component.userMessage = 'open vs code';
    component.submitCommand();

    expect(component.errorMessage).toContain('Policy restriction');
  });

  it('should clear error and response when clearResponse is called', () => {
    const fixture = TestBed.createComponent(CommandBarComponent);
    const component = fixture.componentInstance;
    component.errorMessage = 'Some error';
    component.lastResponse = { turn_id: 't-1', status: 'success', response: 'Hi' } as any;

    component.clearResponse();

    expect(component.errorMessage).toBeNull();
    expect(component.lastResponse).toBeNull();
  });

  it('should display DEMO MODE badge when demo mode is enabled', () => {
    appStateMock.isDemoMode.mockReturnValue(true);
    const fixture = TestBed.createComponent(CommandBarComponent);
    fixture.detectChanges();

    const demoTag = fixture.nativeElement.querySelector('.demo-tag') as HTMLElement;
    expect(demoTag).toBeTruthy();
    expect(demoTag.textContent).toContain('DEMO MODE');
    const input = fixture.nativeElement.querySelector('.command-input') as HTMLInputElement;
    expect(input.placeholder).toContain('DEMO MODE');
  });
});
