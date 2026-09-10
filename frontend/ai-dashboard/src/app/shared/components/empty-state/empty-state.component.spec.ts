import { TestBed } from '@angular/core/testing';
import { EmptyStateComponent } from './empty-state.component';

describe('EmptyStateComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EmptyStateComponent]
    }).compileComponents();
  });

  it('should display title and message', () => {
    const fixture = TestBed.createComponent(EmptyStateComponent);
    fixture.componentRef.setInput('title', 'No Data Available');
    fixture.componentRef.setInput('message', 'Please sync with backend.');
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.title')?.textContent).toContain('No Data Available');
    expect(el.querySelector('.description')?.textContent).toContain('Please sync with backend.');
  });

  it('should emit action event when action button is clicked', () => {
    const fixture = TestBed.createComponent(EmptyStateComponent);
    fixture.componentRef.setInput('title', 'No Assets');
    fixture.componentRef.setInput('actionText', 'Sync Assets');
    fixture.detectChanges();

    let emitted = false;
    fixture.componentInstance.action.subscribe(() => emitted = true);

    const btn = fixture.nativeElement.querySelector('.action-wrap button') as HTMLButtonElement;
    expect(btn).toBeTruthy();
    btn.click();

    expect(emitted).toBe(true);
  });
});
