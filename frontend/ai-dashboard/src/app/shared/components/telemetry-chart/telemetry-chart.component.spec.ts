import { TestBed } from '@angular/core/testing';
import { TelemetryChartComponent } from './telemetry-chart.component';

describe('TelemetryChartComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TelemetryChartComponent]
    }).compileComponents();
  });

  it('should compute min, max, and latest value accurately', () => {
    const fixture = TestBed.createComponent(TelemetryChartComponent);
    fixture.componentRef.setInput('data', [10, 25, 5, 80, 45]);
    fixture.detectChanges();

    const comp = fixture.componentInstance;
    expect(comp.minValue).toBe(5);
    expect(comp.maxValue).toBe(80);
    expect(comp.latestValue).toBe(45);
  });

  it('should render SVG polyline for the sparkline path', () => {
    const fixture = TestBed.createComponent(TelemetryChartComponent);
    fixture.componentRef.setInput('data', [50, 60, 70]);
    fixture.detectChanges();

    const polyline = fixture.nativeElement.querySelector('polyline');
    expect(polyline).toBeTruthy();
    expect(polyline.getAttribute('points')).toBeTruthy();
  });
});
