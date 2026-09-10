import { TestBed } from '@angular/core/testing';
import { StatusBadgeComponent } from './status-badge.component';

describe('StatusBadgeComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [StatusBadgeComponent]
    }).compileComponents();
  });

  it('should render correct class and label for HEALTHY status', () => {
    const fixture = TestBed.createComponent(StatusBadgeComponent);
    fixture.componentRef.setInput('status', 'HEALTHY');
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.badge-healthy')).toBeTruthy();
    expect(el.textContent).toContain('HEALTHY');
  });

  it('should render correct class and label for CRITICAL status', () => {
    const fixture = TestBed.createComponent(StatusBadgeComponent);
    fixture.componentRef.setInput('status', 'CRITICAL');
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.badge-critical')).toBeTruthy();
    expect(el.textContent).toContain('CRITICAL');
  });

  it('should render online status properly', () => {
    const fixture = TestBed.createComponent(StatusBadgeComponent);
    fixture.componentRef.setInput('status', 'ONLINE');
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.badge-healthy')).toBeTruthy();
    expect(el.textContent).toContain('ONLINE');
  });
});
