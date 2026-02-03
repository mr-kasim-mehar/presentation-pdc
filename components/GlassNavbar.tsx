'use client';
import { useState, useEffect, useRef } from 'react';

type NavItem = {
  id: number;
  label: string;
  index: number;
};

interface GlassNavbarProps {
  items: NavItem[];
  activeIndex: number;
  onItemClick: (index: number, item: NavItem) => void;
}

const GlassNavbar = ({ items, activeIndex, onItemClick }: GlassNavbarProps) => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Refs for tracking elements
  const navRef = useRef<HTMLElement>(null);
  const itemsRef = useRef<(HTMLButtonElement | null)[]>([]);
  const [activeRect, setActiveRect] = useState({ left: 0, width: 0, opacity: 0 });
  const [hoverRect, setHoverRect] = useState({ left: 0, width: 0, opacity: 0 });

  // Update Active Pill Position
  useEffect(() => {
    const updateActivePill = () => {
      const currentItem = itemsRef.current[activeIndex];
      if (currentItem) {
        setActiveRect({
          left: currentItem.offsetLeft,
          width: currentItem.offsetWidth,
          opacity: 1
        });
      }
    };

    // Initial delay to allow layout to settle
    const timeout = setTimeout(updateActivePill, 100);
    window.addEventListener('resize', updateActivePill);

    return () => {
      window.removeEventListener('resize', updateActivePill);
      clearTimeout(timeout);
    };
  }, [activeIndex, items]);

  // Update Hover Pill Position
  useEffect(() => {
    if (hoveredIndex !== null) {
      const hoveredItem = itemsRef.current[hoveredIndex];
      if (hoveredItem) {
        setHoverRect({
          left: hoveredItem.offsetLeft,
          width: hoveredItem.offsetWidth,
          opacity: 1
        });
      }
    } else {
      setHoverRect(prev => ({ ...prev, opacity: 0 }));
    }
  }, [hoveredIndex]);


  // Scroll Logic
  useEffect(() => {
    let ticking = false;
    const handleScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          const container = document.querySelector('[data-scroll-container]') as HTMLElement;
          const scrollY = container ? container.scrollTop : window.scrollY;
          setIsScrolled(scrollY > 20);
          ticking = false;
        });
        ticking = true;
      }
    };

    const container = document.querySelector('[data-scroll-container]') as HTMLElement;
    if (container) {
      container.addEventListener('scroll', handleScroll, { passive: true });
    }
    window.addEventListener('scroll', handleScroll, { passive: true });

    return () => {
      if (container) {
        container.removeEventListener('scroll', handleScroll);
      }
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <>
      <nav
        ref={navRef}
        className={`fixed left-1/2 z-[120] -translate-x-1/2 transition-all duration-700 ease-[cubic-bezier(0.25,1,0.5,1)] 
            ${isScrolled ? 'top-4 w-[90%] md:w-auto' : 'top-6 w-[95%] md:w-auto'}`}
        aria-label="Main navigation"
      >
        <div
          className={`
                group relative flex items-center justify-between gap-2 p-1.5 
                backdrop-blur-2xl border transition-all duration-700 ease-[cubic-bezier(0.25,1,0.5,1)]
                ${isScrolled
              ? 'rounded-full border-white/10 bg-black/60 shadow-[0_8px_32px_rgb(0,0,0,0.4)]'
              : 'rounded-2xl border-white/5 bg-black/30 shadow-[0_4px_24px_rgb(0,0,0,0.2)]'
            }
            `}
        >
          {/* Logo */}
          <div className="flex items-center gap-3 pl-3 pr-2">
            <div className="relative flex h-9 w-9 items-center justify-center overflow-hidden rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 shadow-lg group-hover:scale-105 transition-transform duration-500">
              <span className="font-mono text-[10px] font-bold text-white">PDC</span>
              <div className="absolute inset-0 bg-white/30 opacity-0 transition-opacity group-hover:opacity-100 mix-blend-overlay" />
            </div>
            <span className={`font-medium tracking-tight text-white transition-all duration-300 hidden sm:block ${isScrolled ? 'opacity-0 w-0 overflow-hidden' : 'opacity-100 w-auto'}`}>
              PDC
            </span>
          </div>

          {/* Desktop Navigation - Sliding Pill */}
          <div className={`hidden md:flex items-center relative rounded-full p-1 transition-all duration-500 ${isScrolled ? 'bg-transparent' : 'bg-white/5'}`}>

            {/* 1. SLIDING ACTIVE PILL (The magic movement) */}
            <div
              className="absolute top-1 bottom-1 rounded-full bg-gradient-to-r from-blue-600 to-indigo-600 shadow-[0_0_20px_rgba(59,130,246,0.5)] z-0 transition-[left,width,opacity] duration-500 ease-[cubic-bezier(0.22,1,0.36,1)]"
              style={{
                left: activeRect.left,
                width: activeRect.width,
                opacity: activeRect.opacity
              }}
            />

            {/* 2. HOVER SPOTLIGHT (Subtle interaction) */}
            <div
              className="absolute top-1 bottom-1 rounded-full bg-white/10 z-0 transition-[left,width,opacity] duration-300 ease-out pointer-events-none"
              style={{
                left: hoverRect.left,
                width: hoverRect.width,
                opacity: hoverRect.opacity
              }}
            />

            {items.map((item, idx) => {
              const isActive = item.index === activeIndex;
              return (
                <button
                  key={item.id}
                  ref={(el) => { itemsRef.current[item.index] = el; }}
                  type="button"
                  onClick={() => onItemClick(item.index, item)}
                  onMouseEnter={() => setHoveredIndex(item.index)}
                  onMouseLeave={() => setHoveredIndex(null)}
                  className={`
                                relative px-5 py-2.5 text-xs font-semibold tracking-wide uppercase transition-colors duration-300 z-10
                                ${isActive ? 'text-white' : 'text-zinc-400 hover:text-white'}
                            `}
                >
                  {/* Text Effect */}
                  <span className="relative z-20">
                    {item.label.split('. ')[1] || item.label}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Mobile Toggle */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-white/5 text-white transition-all hover:bg-white/10 active:scale-95 md:hidden"
          >
            <div className={`relative h-2.5 w-4 transition-all duration-500 ease-[cubic-bezier(0.25,1,0.5,1)] ${isMobileMenuOpen ? 'rotate-180' : ''}`}>
              <span className={`absolute left-0 top-0 h-[2px] w-full rounded-full bg-current transition-all duration-300 ${isMobileMenuOpen ? 'top-1.5 rotate-45' : ''}`} />
              <span className={`absolute left-0 top-1.5 h-[2px] w-full rounded-full bg-current transition-all duration-300 ${isMobileMenuOpen ? 'opacity-0' : ''}`} />
              <span className={`absolute left-0 bottom-0 h-[2px] w-full rounded-full bg-current transition-all duration-300 ${isMobileMenuOpen ? 'bottom-1.5 -rotate-45' : ''}`} />
            </div>
          </button>
        </div>
      </nav>

      {/* Mobile Menu - Staggered Glass Cards */}
      <div
        className={`fixed inset-0 z-[115] bg-black/90 backdrop-blur-3xl transition-all duration-700 ease-[cubic-bezier(0.22,1,0.36,1)] md:hidden 
            ${isMobileMenuOpen ? 'opacity-100 visible clip-circle-in' : 'opacity-0 invisible clip-circle-out'}`}
        style={{
          clipPath: isMobileMenuOpen ? 'circle(150% at 100% 0)' : 'circle(0% at 100% 0)',
          transition: 'all 0.7s cubic-bezier(0.22, 1, 0.36, 1)'
        }}
      >
        <div className="flex h-full flex-col justify-center px-8 pb-10">
          <div className="flex flex-col space-y-3">
            {items.map((item, idx) => {
              const isActive = item.index === activeIndex;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    onItemClick(item.index, item);
                    setIsMobileMenuOpen(false);
                  }}
                  style={{
                    transitionDelay: isMobileMenuOpen ? `${100 + idx * 50}ms` : '0ms',
                    opacity: isMobileMenuOpen ? 1 : 0,
                    transform: isMobileMenuOpen ? 'translateY(0)' : 'translateY(20px)'
                  }}
                  className={`
                                    group relative flex items-center justify-between overflow-hidden rounded-2xl border p-5 text-left transition-all duration-500 ease-out
                                    ${isActive
                      ? 'border-blue-500/50 bg-gradient-to-r from-blue-900/40 to-indigo-900/40'
                      : 'border-white/5 bg-white/5 hover:bg-white/10'
                    }
                                `}
                >
                  <span className={`text-xl font-bold tracking-tight transition-colors ${isActive ? 'text-white' : 'text-zinc-500 group-hover:text-zinc-300'}`}>
                    {item.label}
                  </span>

                  {isActive && (
                    <div className="absolute right-4 h-2 w-2 rounded-full bg-blue-400 shadow-[0_0_10px_#60A5FA]" />
                  )}
                </button>
              );
            })}
          </div>

          <button
            onClick={() => setIsMobileMenuOpen(false)}
            className="mt-10 self-center rounded-full bg-white/10 px-8 py-3 text-sm font-semibold text-white/50 hover:text-white hover:bg-white/20 transition-all active:scale-95"
          >
            Close Menu
          </button>
        </div>
      </div>
    </>
  );
};

export default GlassNavbar;
