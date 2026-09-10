import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'command-center',
    pathMatch: 'full'
  },
  {
    path: 'command-center',
    loadComponent: () => import('./features/command-center/command-center.component').then(m => m.CommandCenterComponent)
  },
  {
    path: 'products',
    loadComponent: () => import('./features/products/products.component').then(m => m.ProductsComponent)
  },
  {
    path: 'products/:id',
    loadComponent: () => import('./features/products/product-detail/product-detail.component').then(m => m.ProductDetailComponent)
  },
  {
    path: 'situations',
    loadComponent: () => import('./features/situations/situations.component').then(m => m.SituationsComponent)
  },
  {
    path: 'missions',
    loadComponent: () => import('./features/missions/missions.component').then(m => m.MissionsComponent)
  },
  {
    path: 'world',
    loadComponent: () => import('./features/world-state/world-state.component').then(m => m.WorldStateComponent)
  },
  {
    path: 'perception',
    loadComponent: () => import('./features/perception/perception.component').then(m => m.PerceptionComponent)
  },
  {
    path: 'events',
    loadComponent: () => import('./features/events/events.component').then(m => m.EventsComponent)
  },
  {
    path: 'traces',
    loadComponent: () => import('./features/traces/traces.component').then(m => m.TracesComponent)
  },
  {
    path: 'simulation',
    loadComponent: () => import('./features/simulation/simulation.component').then(m => m.SimulationComponent)
  },
  {
    path: 'settings',
    loadComponent: () => import('./features/settings/settings.component').then(m => m.SettingsComponent)
  },
  {
    path: '**',
    redirectTo: 'command-center'
  }
];
