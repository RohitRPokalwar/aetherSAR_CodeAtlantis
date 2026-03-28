import { createBrowserRouter } from 'react-router';
import AetherFlowHero from './components/ui/aether-flow-hero';
import { SARDashboard } from './components/sar/sar-dashboard';

export const router = createBrowserRouter([
  {
    path: '/',
    Component: AetherFlowHero,
  },
  {
    path: '/dashboard',
    Component: SARDashboard,
  },
]);
