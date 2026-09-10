import { TestBed } from '@angular/core/testing';
import { ConfirmationDialogComponent } from './confirmation-dialog.component';

describe('ConfirmationDialogComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ConfirmationDialogComponent]
    }).compileComponents();
  });

  it('should not show backdrop when isOpen is false', () => {
    const fixture = TestBed.createComponent(ConfirmationDialogComponent);
    fixture.componentRef.setInput('isOpen', false);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.dialog-backdrop')).toBeNull();
  });

  it('should show dialog and emit confirm when confirmed', () => {
    const fixture = TestBed.createComponent(ConfirmationDialogComponent);
    fixture.componentRef.setInput('isOpen', true);
    fixture.componentRef.setInput('title', 'Authorize Action');
    fixture.componentRef.setInput('message', 'Are you sure?');
    fixture.detectChanges();

    let confirmedEmitted = false;
    fixture.componentInstance.confirm.subscribe(() => {
      confirmedEmitted = true;
    });

    const confirmBtn = fixture.nativeElement.querySelector('.btn-primary') as HTMLButtonElement;
    expect(confirmBtn).toBeTruthy();
    confirmBtn.click();

    expect(confirmedEmitted).toBe(true);
  });
});
