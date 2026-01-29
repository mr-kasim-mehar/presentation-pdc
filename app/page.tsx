'use client';
import dynamic from "next/dynamic";
import Image from "next/image";
import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import GlassNavbar from "@/components/GlassNavbar";

const LightRays = dynamic(() => import("@/components/LightRays"), {
  ssr: false,
});

// Updated Slide Data with Short Titles for Navbar
const slides = [
  { id: 0, title: "Home", name: "Project Title & Team" },
  { id: 1, title: "Strategy", name: "Implementation Strategy" },
  { id: 2, title: "Module 1", name: "Secure Authentication" },
  { id: 3, title: "Module 2", name: "Researcher Workspace" },
  { id: 4, title: "Dashboard", name: "Researcher Dashboard" },
  { id: 5, title: "Module 3", name: "Reporting Engine" },
  { id: 6, title: "Module 4", name: "Triage Management" },
  { id: 7, title: "Module 5", name: "Admin Control" },
  { id: 8, title: "Module 6", name: "Company Programs" },
  { id: 9, title: "Backend", name: "Backend Architecture" },
  { id: 10, title: "Frontend", name: "Frontend & Data" },
  { id: 11, title: "Demo 1", name: "Reporting Flow Demo" },
  { id: 12, title: "Demo 2", name: "Management Flow Demo" },
  { id: 13, title: "Roadmap", name: "Roadmap to Capstone 2" },
  { id: 14, title: "Future", name: "Conclusion & Future" },
];

export default function Home() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [selectedImage, setSelectedImage] = useState<{ src: string; alt: string } | null>(null);
  const slideRefs = useRef<(HTMLDivElement | null)[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);
  const isScrolling = useRef(false);

  // Memoize nav items
  const navItems = useMemo(() => {
    return slides.map((slide, index) => ({
      id: slide.id,
      label: `${index + 1}. ${slide.title}`,
      index: index
    }));
  }, []);

  // GSAP animation setup
  useEffect(() => {
    const initGSAP = async () => {
      const gsap = (await import("gsap")).default;
      const ScrollToPlugin = (await import("gsap/ScrollToPlugin")).default;
      const ScrollTrigger = (await import("gsap/ScrollTrigger")).default;

      gsap.registerPlugin(ScrollToPlugin, ScrollTrigger);

      slideRefs.current.forEach((slide, index) => {
        if (!slide) return;

        const heading = slide.querySelector('h1');
        const subContent = slide.querySelectorAll('p, h2, ul, ol, .content-box');

        if (heading) {
          gsap.fromTo(heading,
            { y: 50, opacity: 0 },
            {
              y: 0,
              opacity: 1,
              duration: 1,
              ease: "power3.out",
              scrollTrigger: {
                trigger: slide,
                start: "top 80%",
                scroller: containerRef.current,
                toggleActions: "play reverse play reverse"
              }
            }
          );
        }

        if (subContent.length > 0) {
          gsap.fromTo(subContent,
            { y: 30, opacity: 0 },
            {
              y: 0,
              opacity: 1,
              duration: 0.8,
              stagger: 0.1,
              ease: "power2.out",
              scrollTrigger: {
                trigger: slide,
                start: "top 75%",
                scroller: containerRef.current,
                toggleActions: "play reverse play reverse"
              }
            }
          );
        }
      });
    };

    initGSAP();
  }, []);

  const scrollToSlide = useCallback(async (index: number) => {
    if (isScrolling.current) return;

    // Ensure index is valid
    if (index < 0 || index >= slides.length) return;

    const slideElement = slideRefs.current[index];
    if (slideElement && containerRef.current) {
      isScrolling.current = true;
      const gsap = (await import("gsap")).default;

      gsap.to(containerRef.current, {
        scrollTo: { y: slideElement, autoKill: false },
        duration: 1.2,
        ease: "power4.inOut",
        onComplete: () => {
          isScrolling.current = false;
          setCurrentSlide(index);
        }
      });

      setCurrentSlide(index);
    }
  }, []);

  const handleNavClick = useCallback((index: number, item: any) => {
    scrollToSlide(index);
  }, [scrollToSlide]);

  // Scroll detection
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      if (isScrolling.current) return;

      const scrollTop = container.scrollTop;
      const windowHeight = window.innerHeight;

      for (let i = 0; i < slideRefs.current.length; i++) {
        const slide = slideRefs.current[i];
        if (slide) {
          const slideTop = slide.offsetTop;
          const slideBottom = slideTop + slide.offsetHeight;

          if (scrollTop + windowHeight / 2 >= slideTop && scrollTop + windowHeight / 2 < slideBottom) {
            if (currentSlide !== i) {
              setCurrentSlide(i);
            }
            break;
          }
        }
      }
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, [currentSlide]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && selectedImage) {
        setSelectedImage(null);
        return;
      }

      if (selectedImage) return;

      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        const nextSlide = Math.min(currentSlide + 1, slides.length - 1);
        scrollToSlide(nextSlide);
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        const prevSlide = Math.max(currentSlide - 1, 0);
        scrollToSlide(prevSlide);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [currentSlide, scrollToSlide, selectedImage]);

  // Helper for glass cards
  const GlassCard = ({ children, className = "" }: { children: React.ReactNode, className?: string }) => (
    <div className={`backdrop-blur-lg bg-black/30 p-4 sm:p-6 md:p-8 rounded-xl border border-white/10 shadow-2xl ${className}`}>
      {children}
    </div>
  );

  return (
    <div className="relative min-h-screen w-full" style={{ backgroundColor: '#000000' }}>
      {/* Background */}
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', zIndex: 1 }}>
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
          <LightRays
            raysOrigin="top-center"
            raysColor="#ffffff"
            raysSpeed={1}
            lightSpread={0.5}
            rayLength={3}
            followMouse={true}
            mouseInfluence={0.1}
            noiseAmount={0}
            distortion={0}
            className="custom-rays"
            pulsating={false}
            fadeDistance={1}
            saturation={1}
          />
        </div>
      </div>

      {/* Navigation */}
      <GlassNavbar items={navItems} activeIndex={currentSlide} onItemClick={handleNavClick} />

      {/* Pages */}
      <div
        ref={containerRef}
        data-scroll-container
        className="relative overflow-y-auto overflow-x-hidden h-screen snap-y snap-mandatory scroll-smooth"
        style={{ scrollBehavior: 'smooth', zIndex: 10 }}
      >

        {/* Slide 1: Project Title & Team */}
        <div ref={(el) => { slideRefs.current[0] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-4xl w-full text-center space-y-8">
            <GlassCard>
              <h1 className="text-4xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 mb-4">
                BugChase
              </h1>
              <p className="text-xl md:text-2xl text-blue-200 font-medium mb-6">
                A Specialized Bug Bounty Platform for Pakistan
              </p>
              <div className="h-px w-32 bg-white/20 mx-auto my-6" />
              <p className="text-lg text-gray-300 italic">
                "Bridging the gap between local companies and ethical hackers."
              </p>
            </GlassCard>

            <GlassCard className="text-left">
              <h3 className="text-xl font-bold text-white mb-4 border-b border-white/10 pb-2">Team Members</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  <span className="text-gray-200">Muhammad Qasim</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  <span className="text-gray-200">Shahzaib Ahmad</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  <span className="text-gray-200">Shahzaib</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  <span className="text-gray-200">Tauseef Ahmad</span>
                </div>
              </div>
              <div className="mt-6 pt-4 border-t border-white/10">
                <p className="text-gray-300">
                  <span className="font-semibold text-white">Supervisor:</span> Madam Sumbal Fatima
                </p>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 2: Implementation Strategy */}
        <div ref={(el) => { slideRefs.current[1] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center drop-shadow-lg">Implementation Strategy</h1>
            <GlassCard className="space-y-6">
              <div className="content-box">
                <h3 className="text-2xl font-bold text-blue-400 mb-2">Milestone Goal</h3>
                <p className="text-gray-300 text-lg">40% implementation of core system workflows.</p>
              </div>
              <div className="content-box">
                <h3 className="text-xl font-semibold text-white mb-2">Approach</h3>
                <p className="text-gray-300">Focused on the <span className="text-blue-300">"Researcher-to-Company"</span> lifecycle, ensuring a functional end-to-end path for vulnerability reporting.</p>
              </div>
              <div className="content-box">
                <h3 className="text-xl font-semibold text-white mb-2">Current Status</h3>
                <ul className="list-disc list-inside text-gray-300 space-y-1">
                  <li>Core security architecture</li>
                  <li>Profile management systems</li>
                  <li>Reporting engine fully operational</li>
                </ul>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 3: Module 1 – Secure Authentication */}
        <div ref={(el) => { slideRefs.current[2] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Module 1: Secure Authentication</h1>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <GlassCard className="content-box">
                <div className="h-12 w-12 rounded-lg bg-blue-500/20 flex items-center justify-center mb-4">
                  <svg className="w-6 h-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Registration</h3>
                <p className="text-gray-400">Multi-role signup flow for Researchers, Companies, and Administrators.</p>
              </GlassCard>

              <GlassCard className="content-box">
                <div className="h-12 w-12 rounded-lg bg-purple-500/20 flex items-center justify-center mb-4">
                  <svg className="w-6 h-6 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Email Verification</h3>
                <p className="text-gray-400">Integrated OTP (One-Time Password) system for identity verification.</p>
              </GlassCard>

              <GlassCard className="content-box">
                <div className="h-12 w-12 rounded-lg bg-green-500/20 flex items-center justify-center mb-4">
                  <svg className="w-6 h-6 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Session Security</h3>
                <p className="text-gray-400">Industry-standard JWT (JSON Web Tokens) for secure, stateless user sessions.</p>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 4: Module 2 – Researcher Workspace */}
        <div ref={(el) => { slideRefs.current[3] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Module 2: Researcher Workspace</h1>
            <GlassCard className="space-y-6">
              <div className="content-box border-l-4 border-blue-500 pl-4 py-2 bg-white/5 rounded-r-lg">
                <h3 className="text-xl font-bold text-white">Profile Management</h3>
                <p className="text-gray-300">Comprehensive profiles including technical skills, experience, and bios.</p>
              </div>
              <div className="content-box border-l-4 border-indigo-500 pl-4 py-2 bg-white/5 rounded-r-lg">
                <h3 className="text-xl font-bold text-white">Professional Presence</h3>
                <p className="text-gray-300">Integration of GitHub & LinkedIn to build researcher credibility.</p>
              </div>
              <div className="content-box border-l-4 border-cyan-500 pl-4 py-2 bg-white/5 rounded-r-lg">
                <h3 className="text-xl font-bold text-white">Portfolio System</h3>
                <p className="text-gray-300">Automated generation of public portfolio pages showcasing history and successes.</p>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 5: Researcher Dashboard */}
        <div ref={(el) => { slideRefs.current[4] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Researcher Dashboard</h1>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <GlassCard className="content-box flex flex-col justify-center items-center text-center h-full">
                <div className="w-20 h-20 bg-blue-600 rounded-full flex items-center justify-center mb-6 shadow-lg shadow-blue-500/50">
                  <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">Real-time Tracking</h3>
                <p className="text-gray-300">Centralized dashboard for researchers to monitor the live status of their submissions.</p>
              </GlassCard>
              <GlassCard className="content-box flex flex-col justify-center items-center text-center h-full">
                <div className="w-20 h-20 bg-emerald-600 rounded-full flex items-center justify-center mb-6 shadow-lg shadow-emerald-500/50">
                  <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">Visibility Control</h3>
                <p className="text-gray-300">Privacy toggles allowing researchers to choose between public and private visibility.</p>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 6: Module 3 – Reporting Engine */}
        <div ref={(el) => { slideRefs.current[5] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Module 3: Vulnerability Reporting</h1>
            <GlassCard className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="content-box bg-white/5 p-4 rounded-xl text-center">
                  <h4 className="text-blue-400 font-bold text-lg mb-2">Standardized Form</h4>
                  <p className="text-sm text-gray-300">Captures critical data: Title, Category, Description.</p>
                </div>
                <div className="content-box bg-white/5 p-4 rounded-xl text-center">
                  <h4 className="text-blue-400 font-bold text-lg mb-2">Technical Evidence</h4>
                  <p className="text-sm text-gray-300">File upload system supporting Images and Video POCs.</p>
                </div>
                <div className="content-box bg-white/5 p-4 rounded-xl text-center">
                  <h4 className="text-blue-400 font-bold text-lg mb-2">CWE Mapping</h4>
                  <p className="text-sm text-gray-300">Integrated CWE classification for standardized tagging.</p>
                </div>
              </div>
              {/* Visual Mockup of Form */}
              <div className="bg-black/40 rounded-lg p-4 border border-white/5 mt-4">
                <div className="h-4 w-1/3 bg-white/10 rounded mb-4"></div>
                <div className="h-10 w-full bg-white/5 rounded mb-4 border border-white/10"></div>
                <div className="h-32 w-full bg-white/5 rounded border border-white/10 flex items-center justify-center text-gray-500 text-sm">Description Area</div>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 7: Module 4 – Triage Management */}
        <div ref={(el) => { slideRefs.current[6] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Module 4: Triage & Status</h1>
            <div className="space-y-4">
              <GlassCard className="flex items-start space-x-4 content-box transform hover:scale-[1.02] transition-transform">
                <div className="mt-1 bg-yellow-500/20 p-2 rounded-lg"><span className="text-2xl">🔄</span></div>
                <div>
                  <h3 className="text-xl font-bold text-white">Status Workflow</h3>
                  <p className="text-gray-300">Logic-driven state machine (Pending → Needs Info → Resolved).</p>
                </div>
              </GlassCard>
              <GlassCard className="flex items-start space-x-4 content-box transform hover:scale-[1.02] transition-transform">
                <div className="mt-1 bg-red-500/20 p-2 rounded-lg"><span className="text-2xl">🔒</span></div>
                <div>
                  <h3 className="text-xl font-bold text-white">Collision Prevention</h3>
                  <p className="text-gray-300">"Lock" mechanism prevents multiple triagers editing the same report.</p>
                </div>
              </GlassCard>
              <GlassCard className="flex items-start space-x-4 content-box transform hover:scale-[1.02] transition-transform">
                <div className="mt-1 bg-green-500/20 p-2 rounded-lg"><span className="text-2xl">✅</span></div>
                <div>
                  <h3 className="text-xl font-bold text-white">Resolution Path</h3>
                  <p className="text-gray-300">Mark reports as Duplicate, Spam, or Out-of-Scope.</p>
                </div>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 8: Module 5 – Administrative Control */}
        <div ref={(el) => { slideRefs.current[7] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Module 5: Admin Control</h1>
            <GlassCard className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
              <div className="content-box p-4 border border-white/10 rounded-xl hover:bg-white/5 transition-colors">
                <h3 className="text-white font-bold text-lg mb-2">User Moderation</h3>
                <p className="text-gray-400 text-sm">Suspend or ban users violating policies.</p>
              </div>
              <div className="content-box p-4 border border-white/10 rounded-xl hover:bg-white/5 transition-colors">
                <h3 className="text-white font-bold text-lg mb-2">Staff Onboarding</h3>
                <p className="text-gray-400 text-sm">Secure workflow to invite technical Triagers.</p>
              </div>
              <div className="content-box p-4 border border-white/10 rounded-xl hover:bg-white/5 transition-colors">
                <h3 className="text-white font-bold text-lg mb-2">RBAC</h3>
                <p className="text-gray-400 text-sm">Role-based access control for staff.</p>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 9: Module 6 – Company & Program Basics */}
        <div ref={(el) => { slideRefs.current[8] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Module 6: Company Programs</h1>
            <GlassCard className="space-y-8">
              <div className="content-box">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-2xl font-bold text-white">Program Directory</h3>
                  <span className="bg-blue-500/20 text-blue-300 px-3 py-1 rounded-full text-sm">Searchable</span>
                </div>
                <p className="text-gray-300">Directory of security programs categorized by industry (Fintech, Crypto, etc).</p>
              </div>
              <div className="content-box">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-2xl font-bold text-white">Scope Definition</h3>
                  <span className="bg-purple-500/20 text-purple-300 px-3 py-1 rounded-full text-sm">Guide</span>
                </div>
                <p className="text-gray-300">Functionality to define "Out-of-Scope" assets for researchers.</p>
              </div>
              <div className="content-box">
                <h3 className="text-2xl font-bold text-white mb-2">Public Programs</h3>
                <p className="text-gray-300">Launch public-facing vulnerability disclosure programs.</p>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 10: Technical Architecture (Backend) */}
        <div ref={(el) => { slideRefs.current[9] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-4xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Technical Architecture (Backend)</h1>
            <GlassCard className="space-y-6">
              <div className="content-box flex items-center p-4 bg-white/5 rounded-xl">
                <div className="w-16 h-16 bg-green-500/20 rounded-lg flex items-center justify-center mr-6 text-2xl font-bold text-green-400">Node</div>
                <div>
                  <h3 className="text-xl font-bold text-white">Node.js + Express.js</h3>
                  <p className="text-gray-300">High-performance API backbone.</p>
                </div>
              </div>
              <div className="content-box flex items-center p-4 bg-white/5 rounded-xl">
                <div className="w-16 h-16 bg-blue-500/20 rounded-lg flex items-center justify-center mr-6 text-2xl font-bold text-blue-400">Logic</div>
                <div>
                  <h3 className="text-xl font-bold text-white">Logic Layer</h3>
                  <p className="text-gray-300">Centralized handling of report states and permissions.</p>
                </div>
              </div>
              <div className="content-box flex items-center p-4 bg-white/5 rounded-xl">
                <div className="w-16 h-16 bg-red-500/20 rounded-lg flex items-center justify-center mr-6 text-2xl font-bold text-red-400">Sec</div>
                <div>
                  <h3 className="text-xl font-bold text-white">Security</h3>
                  <p className="text-gray-300">Advanced hashing algorithms & secure token authentication.</p>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 11: Technical Architecture (Frontend) */}
        <div ref={(el) => { slideRefs.current[10] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-4xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Technical Architecture (Frontend)</h1>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <GlassCard className="content-box text-center">
                <div className="text-6xl mb-4">⚛️</div>
                <h3 className="text-2xl font-bold text-white mb-2">React.js</h3>
                <p className="text-gray-300">Responsive, real-time dashboards for Researchers, Companies, and Admins.</p>
              </GlassCard>
              <GlassCard className="content-box text-center">
                <div className="text-6xl mb-4">🍃</div>
                <h3 className="text-2xl font-bold text-white mb-2">MongoDB</h3>
                <p className="text-gray-300">NoSQL database for flexible storage of diverse vulnerability report metadata.</p>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 12: Use Case Demo - Reporting */}
        <div ref={(el) => { slideRefs.current[11] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Demo: The Reporting Flow</h1>
            <GlassCard className="relative overflow-hidden">
              {/* Flow Line */}
              <div className="absolute top-1/2 left-10 right-10 h-1 bg-white/10 -translate-y-1/2 hidden md:block" />

              <div className="relative grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="content-box bg-black/40 p-6 rounded-xl border border-white/10 z-10 text-center">
                  <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold mx-auto mb-4">1</div>
                  <h4 className="font-bold text-white mb-2">Register</h4>
                  <p className="text-sm text-gray-400">Identity verification via OTP.</p>
                </div>
                <div className="content-box bg-black/40 p-6 rounded-xl border border-white/10 z-10 text-center">
                  <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold mx-auto mb-4">2</div>
                  <h4 className="font-bold text-white mb-2">Submit</h4>
                  <p className="text-sm text-gray-400">Technical bug report with video evidence.</p>
                </div>
                <div className="content-box bg-black/40 p-6 rounded-xl border border-white/10 z-10 text-center">
                  <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold mx-auto mb-4">3</div>
                  <h4 className="font-bold text-white mb-2">Route</h4>
                  <p className="text-sm text-gray-400">System routes to Triager queue.</p>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 13: Use Case Demo - Management */}
        <div ref={(el) => { slideRefs.current[12] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Demo: Management Flow</h1>
            <div className="space-y-4">
              <GlassCard className="content-box flex items-center space-x-4">
                <div className="font-mono text-blue-400 text-xl font-bold">01</div>
                <div>
                  <h4 className="font-bold text-white">Admin Onboarding</h4>
                  <p className="text-gray-400">Admin onboards new Triager based on expertise.</p>
                </div>
              </GlassCard>
              <GlassCard className="content-box flex items-center space-x-4">
                <div className="font-mono text-blue-400 text-xl font-bold">02</div>
                <div>
                  <h4 className="font-bold text-white">Triage Action</h4>
                  <p className="text-gray-400">Triager claims log, locks it, and requests info.</p>
                </div>
              </GlassCard>
              <GlassCard className="content-box flex items-center space-x-4">
                <div className="font-mono text-blue-400 text-xl font-bold">03</div>
                <div>
                  <h4 className="font-bold text-white">Monitoring</h4>
                  <p className="text-gray-400">Admin monitors platform for policy violations.</p>
                </div>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 14: Roadmap */}
        <div ref={(el) => { slideRefs.current[13] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Roadmap to Capstone 2 (Remaining 60%)</h1>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <GlassCard className="content-box border-t-4 border-blue-500">
                <h3 className="text-xl font-bold text-white mb-4">AI & Automation</h3>
                <p className="text-gray-300">AI-driven CVSS scoring and automated duplicate detection.</p>
              </GlassCard>
              <GlassCard className="content-box border-t-4 border-purple-500">
                <h3 className="text-xl font-bold text-white mb-4">Kali Microservices</h3>
                <p className="text-gray-300">Integration of automated subdomain and port discovery tools.</p>
              </GlassCard>
              <GlassCard className="content-box border-t-4 border-green-500">
                <h3 className="text-xl font-bold text-white mb-4">Financial Integration</h3>
                <p className="text-gray-300">Escrow wallet system and local payment gateway support.</p>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 15: Conclusion */}
        <div ref={(el) => { slideRefs.current[14] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-4xl w-full text-center">
            <h1 className="text-4xl md:text-6xl font-bold text-white mb-8">Future Outlook</h1>
            <GlassCard className="space-y-8 py-12">
              <div className="content-box">
                <p className="text-2xl text-white font-medium mb-2">Milestone Achieved</p>
                <p className="text-gray-400">Essential bug bounty ecosystem foundation is live.</p>
              </div>
              <div className="w-1/2 mx-auto h-px bg-white/10" />
              <div className="content-box">
                <p className="text-2xl text-white font-medium mb-2">Next Focus</p>
                <p className="text-gray-400">Shifting to AI-enhanced automation and financial security.</p>
              </div>
              <div className="w-1/2 mx-auto h-px bg-white/10" />
              <div className="content-box">
                <p className="text-xl text-blue-300 italic">"Providing a world-class security platform tailored for the Pakistani market."</p>
              </div>
            </GlassCard>
          </div>
        </div>

      </div>

      {/* Image Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm"
          onClick={() => setSelectedImage(null)}
          style={{ zIndex: 9999 }}
        >
          <button
            onClick={(e) => { e.stopPropagation(); setSelectedImage(null); }}
            className="absolute top-4 right-4 text-white hover:text-blue-400 p-2 rounded-full bg-black/50"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          <div className="relative max-w-[95vw] max-h-[95vh] p-4" onClick={(e) => e.stopPropagation()}>
            <Image src={selectedImage.src} alt={selectedImage.alt} width={1200} height={800} className="max-w-full max-h-[95vh] rounded-lg shadow-2xl object-contain" />
          </div>
        </div>
      )}
    </div>
  );
}