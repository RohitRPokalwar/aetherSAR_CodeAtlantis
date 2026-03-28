import React from 'react';
import { LayoutDashboard, Crosshair, Bell, FileText, Upload, MapPin, Image as ImageIcon } from 'lucide-react';

export type PageId = 'dashboard' | 'detection' | 'alerts' | 'reports' | 'location' | 'gallery';

interface NavSidebarProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  onOpenUpload: () => void;
  darkMode: boolean;
}

const NAV_ITEMS: { id: PageId; icon: React.ReactNode; label: string }[] = [
  { id: 'dashboard', icon: <LayoutDashboard className="h-5 w-5" />, label: 'Dashboard' },
  { id: 'detection', icon: <Crosshair className="h-5 w-5" />, label: 'Detection' },
  { id: 'location', icon: <MapPin className="h-5 w-5" />, label: 'Ship Location' },
  { id: 'alerts', icon: <Bell className="h-5 w-5" />, label: 'Alerts' },
  { id: 'reports', icon: <FileText className="h-5 w-5" />, label: 'Reports' },
  { id: 'gallery', icon: <ImageIcon className="h-5 w-5" />, label: 'Gallery' },
];

export const NavSidebar: React.FC<NavSidebarProps> = ({ activePage, onNavigate, onOpenUpload, darkMode }) => {
  const bg = darkMode ? '#08121e' : '#f1f5f9';
  const borderC = darkMode ? '#1a2d45' : '#e2e8f0';
  const inactiveColor = darkMode ? '#94a3b8' : '#64748b';
  const hoverBg = darkMode ? 'rgba(0,200,255,0.06)' : 'rgba(0,100,200,0.06)';

  return (
    <aside
      className="flex-none flex flex-col items-center h-full py-3 gap-1"
      style={{ width: '60px', minWidth: '60px', background: bg, borderRight: `1px solid ${borderC}` }}
    >
      {NAV_ITEMS.map(({ id, icon, label }) => {
        const isActive = activePage === id;
        return (
          <button
            key={id}
            onClick={() => onNavigate(id)}
            title={label}
            className="relative w-10 h-10 flex items-center justify-center rounded-lg transition-all duration-200"
            style={{
              color: isActive ? '#00c8ff' : inactiveColor,
              background: isActive ? 'rgba(0,200,255,0.1)' : 'transparent',
            }}
            onMouseEnter={e => {
              if (!isActive) {
                e.currentTarget.style.background = hoverBg;
                e.currentTarget.style.color = '#00c8ff';
              }
            }}
            onMouseLeave={e => {
              if (!isActive) {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = inactiveColor;
              }
            }}
          >
            {/* Active indicator bar */}
            {isActive && (
              <div
                className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r"
                style={{ background: '#00c8ff', boxShadow: '0 0 8px rgba(0,200,255,0.5)' }}
              />
            )}
            {icon}
          </button>
        );
      })}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Upload button */}
      <button
        onClick={onOpenUpload}
        title="Upload Imagery"
        className="w-10 h-10 flex items-center justify-center rounded-lg transition-all duration-200 mb-2"
        style={{
          color: '#00c8ff',
          background: 'rgba(0,200,255,0.08)',
          border: '1px solid rgba(0,200,255,0.2)',
        }}
        onMouseEnter={e => { e.currentTarget.style.background = 'rgba(0,200,255,0.18)'; }}
        onMouseLeave={e => { e.currentTarget.style.background = 'rgba(0,200,255,0.08)'; }}
      >
        <Upload className="h-4 w-4" />
      </button>
    </aside>
  );
};
