import { Component, inject, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, RouterLink, RouterLinkActive } from '@angular/router';
import { AppStateService } from './core/state/app-state.service';
import { StatusBadgeComponent } from './shared/components/status-badge/status-badge.component';
import { CommandBarComponent } from './shared/components/command-bar/command-bar.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    RouterLink,
    RouterLinkActive,
    StatusBadgeComponent,
    CommandBarComponent,
  ],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App implements OnInit, OnDestroy {
  public appState = inject(AppStateService);
  public currentTime = '';
  private timer: any = null;

  public navItems = [
    {
      label: 'Command Center',
      route: '/command-center',
      icon: 'grid'
    },
    {
      label: 'Autonomous Fleet',
      route: '/products',
      icon: 'cpu',
      badge: () => this.appState.onlineProductsCount() + '/' + this.appState.products().length
    },
    {
      label: 'Situation Intel',
      route: '/situations',
      icon: 'alert-triangle',
      badge: () => this.appState.criticalSituationsCount() > 0 ? `${this.appState.criticalSituationsCount()}!` : null
    },
    {
      label: 'Tactical Missions',
      route: '/missions',
      icon: 'target',
      badge: () => this.appState.activeGoalsCount() > 0 ? `${this.appState.activeGoalsCount()}` : null
    },
    {
      label: 'World State Twin',
      route: '/world',
      icon: 'globe'
    },
    {
      label: 'Perception Console',
      route: '/perception',
      icon: 'eye'
    },
    {
      label: 'Live Event Stream',
      route: '/events',
      icon: 'activity'
    },
    {
      label: 'Cognitive Traces',
      route: '/traces',
      icon: 'git-commit'
    },
    {
      label: 'Twin Simulation',
      route: '/simulation',
      icon: 'layers'
    },
    {
      label: 'Settings & Diag',
      route: '/settings',
      icon: 'settings'
    },
  ];

  ngOnInit(): void {
    this.updateClock();
    if (typeof window !== 'undefined') {
      this.timer = setInterval(() => this.updateClock(), 1000);
    }
  }

  ngOnDestroy(): void {
    if (this.timer) {
      clearInterval(this.timer);
    }
  }

  private updateClock(): void {
    const now = new Date();
    this.currentTime = now.toTimeString().split(' ')[0] + ' UTC';
  }
}