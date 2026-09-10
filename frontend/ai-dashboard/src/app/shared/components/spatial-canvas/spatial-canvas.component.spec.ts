import { TestBed } from '@angular/core/testing';
import { SpatialCanvasComponent } from './spatial-canvas.component';
import { Product } from '../../../core/models';

describe('SpatialCanvasComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SpatialCanvasComponent]
    }).compileComponents();
  });

  it('should render SVG canvas elements', () => {
    const fixture = TestBed.createComponent(SpatialCanvasComponent);
    fixture.detectChanges();

    const svg = fixture.nativeElement.querySelector('.radar-svg');
    expect(svg).toBeTruthy();
  });

  it('should map products to distinct radar positions', () => {
    const fixture = TestBed.createComponent(SpatialCanvasComponent);
    const component = fixture.componentInstance;

    const drone: Product = {
      device_id: 'DRONE_01',
      display_name: 'Drone Sentinel',
      device_type: 'drone',
      product_type: 'DRONE',
      product_role: 'AERIAL_SURVEILLANCE',
      contract_version: '1.0',
      is_simulation: true,
      connectivity_status: 'ONLINE',
      health_status: 'HEALTHY',
      capabilities: [],
      registered_at: 100,
    };

    const rover: Product = {
      device_id: 'ROVER_01',
      display_name: 'Rover Vanguard',
      device_type: 'rover',
      product_type: 'ROVER',
      product_role: 'GROUND_PATROL',
      contract_version: '1.0',
      is_simulation: true,
      connectivity_status: 'ONLINE',
      health_status: 'HEALTHY',
      capabilities: [],
      registered_at: 100,
    };

    expect(component.getProductX(drone)).toBe(330);
    expect(component.getProductX(rover)).toBe(370);
    expect(component.getProductColor(drone)).toBe('#00f0ff');
    expect(component.getProductColor(rover)).toBe('#10b981');
  });

  it('should emit productSelected when a product is clicked', () => {
    const fixture = TestBed.createComponent(SpatialCanvasComponent);
    const component = fixture.componentInstance;

    const mockProd: Product = {
      device_id: 'VISION_01',
      display_name: 'Vision Alpha',
      device_type: 'camera',
      product_type: 'VISION',
      product_role: 'FIXED_STATIONARY',
      contract_version: '1.0',
      is_simulation: true,
      connectivity_status: 'ONLINE',
      health_status: 'HEALTHY',
      capabilities: [],
      registered_at: 100,
    };

    fixture.componentRef.setInput('products', [mockProd]);
    fixture.detectChanges();

    let selected: Product | null = null;
    component.productSelected.subscribe(p => selected = p);

    const marker = fixture.nativeElement.querySelector('.product-marker');
    expect(marker).toBeTruthy();
    marker.dispatchEvent(new MouseEvent('click'));

    expect(selected).toEqual(mockProd);
  });
});
