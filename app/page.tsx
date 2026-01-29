'use client';
import dynamic from "next/dynamic";
import Image from "next/image";
import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import GlassNavbar from "@/components/GlassNavbar";

// Lucide Icons Imports
import {
  UserPlus,
  MailCheck,
  ShieldCheck,
  CreditCard,
  ScanFace,
  ShieldAlert,
  BadgeCheck,
  ListChecks,
  Briefcase,
  Link,
  LayoutDashboard,
  Calculator,
  BarChart3,
  FileEdit,
  ImagePlus,
  Tag,
  MessageSquare,
  Send,
  Building2,
  Lock,
  RefreshCw,
  FileText,
  Megaphone,
  UserX,
  UserCog,
  Search,
  Target,
  Activity,
  Server,
  Shield,
  Database,
  Layers,
  Zap,
  Cpu,
  UserCheck,
  Upload,
  MessageCircle
} from "lucide-react";

const LightRays = dynamic(() => import("@/components/LightRays"), {
  ssr: false,
});

// Updated Slide Data with Short Titles for Navbar
const slides = [
  { id: 0, title: "Identity", name: "Project Title & Team Identity" },
  { id: 1, title: "Milestones", name: "Milestone Implementation Overview" },
  { id: 2, title: "Auth", name: "Secure Authentication & Access Control" },
  { id: 3, title: "KYC", name: "Advanced Identity Verification (KYC)" },
  { id: 4, title: "Workspace", name: "Researcher Workspace & Identity" },
  { id: 5, title: "Tools", name: "Researcher Technical Tools" },
  { id: 6, title: "Reporting", name: "Vulnerability Reporting Engine" },
  { id: 7, title: "Chat", name: "Unified Communication Engine (Chat)" },
  { id: 8, title: "Triage", name: "Triage & Validation Workflow" },
  { id: 9, title: "Governance", name: "Administrative Governance & Messaging" },
  { id: 10, title: "Programs", name: "Company Program & Scoping" },
  { id: 11, title: "Backend", name: "Technical Architecture (Backend)" },
  { id: 12, title: "Frontend", name: "Technical Architecture (Frontend)" },
  { id: 13, title: "Demo", name: "Use Case Demo: Verified Reporting" },
  { id: 14, title: "Future", name: "Conclusion & Milestone Roadmap" },
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
                Pakistan’s Specialized Bug Bounty Ecosystem
              </p>
              <div className="h-px w-32 bg-white/20 mx-auto my-6" />
              <p className="text-lg text-gray-300 italic">
                GIFT University, Gujranwala
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
                  <span className="text-gray-200">Shahzab</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  <span className="text-gray-200">Tauseef Ahmad</span>
                </div>
              </div>
              <div className="mt-6 pt-4 border-t border-white/10">
                <p className="text-gray-300">
                  <span className="font-semibold text-white">Supervision:</span> Madam Sumbal Fatima
                </p>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 2: Milestone Implementation Overview */}
        <div ref={(el) => { slideRefs.current[1] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center drop-shadow-lg">Milestone Implementation Overview</h1>
            <GlassCard className="space-y-6">
              <div className="content-box">
                <h3 className="text-2xl font-bold text-blue-400 mb-2">Current Phase</h3>
                <p className="text-gray-300 text-lg">40% functional implementation of the project’s total requirements.</p>
              </div>
              <div className="content-box">
                <h3 className="text-xl font-semibold text-white mb-2">Core Objective</h3>
                <p className="text-gray-300">Establishing the essential "Researcher-to-Company" lifecycle with high-security standards.</p>
              </div>
              <div className="content-box">
                <h3 className="text-xl font-semibold text-white mb-2">Key Achievements</h3>
                <p className="text-gray-300">Successful deployment of identity verification (KYC), tripartite chat, and the technical submission engine.</p>
              </div>
              <div className="content-box">
                <h3 className="text-xl font-semibold text-white mb-2">Functional Modules</h3>
                <p className="text-gray-400 text-sm">Authentication, Identity Management, Reporting Engine, Communication Thread, and Administrative Tools.</p>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 3: Secure Authentication & Access Control */}
        <div ref={(el) => { slideRefs.current[2] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Secure Authentication & Access Control</h1>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <GlassCard className="content-box">
                <div className="h-12 w-12 rounded-lg bg-blue-500/20 flex items-center justify-center mb-4">
                  <UserPlus className="w-6 h-6 text-blue-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Multi-Actor Portal</h3>
                <p className="text-gray-400">Specialized, role-based login and registration flows for Researchers, Companies, and Administrators.</p>
              </GlassCard>

              <GlassCard className="content-box">
                <div className="h-12 w-12 rounded-lg bg-purple-500/20 flex items-center justify-center mb-4">
                  <MailCheck className="w-6 h-6 text-purple-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Identity Protection</h3>
                <p className="text-gray-400">Mandatory email-based OTP (One-Time Password) for account activation and credential recovery.</p>
              </GlassCard>

              <GlassCard className="content-box">
                <div className="h-12 w-12 rounded-lg bg-green-500/20 flex items-center justify-center mb-4">
                  <ShieldCheck className="w-6 h-6 text-green-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Session Management</h3>
                <p className="text-gray-400">Utilization of JSON Web Tokens (JWT) for secure, stateless session handling and API protection.</p>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 4: Advanced Identity Verification (KYC) */}
        <div ref={(el) => { slideRefs.current[3] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Advanced Identity Verification (KYC)</h1>
            <GlassCard className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="content-box border-l-4 border-blue-500 pl-4 py-2 bg-white/5 rounded-r-lg flex items-start gap-3">
                  <CreditCard className="w-8 h-8 text-blue-400 mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="text-xl font-bold text-white">NIC Validation</h3>
                    <p className="text-gray-300">Secure integration for capturing and verifying National Identity Card (NIC) data to ensure platform accountability.</p>
                  </div>
                </div>
                <div className="content-box border-l-4 border-indigo-500 pl-4 py-2 bg-white/5 rounded-r-lg flex items-start gap-3">
                  <ScanFace className="w-8 h-8 text-indigo-400 mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="text-xl font-bold text-white">Liveliness Detection</h3>
                    <p className="text-gray-300">Deployment of the Facenet512 model to perform real-time biometric face matching during onboarding.</p>
                  </div>
                </div>
                <div className="content-box border-l-4 border-cyan-500 pl-4 py-2 bg-white/5 rounded-r-lg flex items-start gap-3">
                  <ShieldAlert className="w-8 h-8 text-cyan-400 mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="text-xl font-bold text-white">Fraud Prevention</h3>
                    <p className="text-gray-300">Advanced algorithms to distinguish between live presence and digital spoofs (photos, videos, or masks).</p>
                  </div>
                </div>
                <div className="content-box border-l-4 border-teal-500 pl-4 py-2 bg-white/5 rounded-r-lg flex items-start gap-3">
                  <BadgeCheck className="w-8 h-8 text-teal-400 mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="text-xl font-bold text-white">Trust Mechanism</h3>
                    <p className="text-gray-300">Automated "Verified" badge assignment upon successful completion of biometric and NIC checks.</p>
                  </div>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 5: Researcher Workspace & Identity */}
        <div ref={(el) => { slideRefs.current[4] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Researcher Workspace & Identity</h1>
            <GlassCard className="space-y-6">
              <div className="content-box flex items-center gap-4">
                <div className="p-3 bg-white/5 rounded-lg">
                  <ListChecks className="w-8 h-8 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white mb-1">Skill Inventory</h3>
                  <p className="text-gray-300">Capability for researchers to manage and display technical specialties like Web, Mobile, or Network security.</p>
                </div>
              </div>
              <div className="content-box flex items-center gap-4">
                <div className="p-3 bg-white/5 rounded-lg">
                  <Briefcase className="w-8 h-8 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white mb-1">Portfolio Generation</h3>
                  <p className="text-gray-300">Automated, professional public pages that showcase a researcher's verified achievements and hall-of-fame entries.</p>
                </div>
              </div>
              <div className="content-box flex items-center gap-4">
                <div className="p-3 bg-white/5 rounded-lg">
                  <Link className="w-8 h-8 text-emerald-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white mb-1">Professional Sync</h3>
                  <p className="text-gray-300">Integration with external platforms like GitHub and LinkedIn to build a holistic researcher persona.</p>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 6: Researcher Technical Tools */}
        <div ref={(el) => { slideRefs.current[5] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Researcher Technical Tools</h1>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <GlassCard className="content-box flex flex-col h-full text-center items-center">
                <LayoutDashboard className="w-12 h-12 text-blue-400 mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Real-time Status Dashboard</h3>
                <p className="text-gray-300 flex-grow">A centralized tracking system for researchers to monitor their reports' progress from "Pending" to "Triaged."</p>
              </GlassCard>
              <GlassCard className="content-box flex flex-col h-full text-center items-center">
                <Calculator className="w-12 h-12 text-indigo-400 mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Built-in CVSS v3.1 Calculator</h3>
                <p className="text-gray-300 flex-grow">A native tool on the researcher side to calculate the severity of a bug before submission.</p>
              </GlassCard>
              <GlassCard className="content-box flex flex-col h-full text-center items-center">
                <BarChart3 className="w-12 h-12 text-red-400 mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Severity Metrics</h3>
                <p className="text-gray-300 flex-grow">Researchers select impact parameters (e.g., Attack Vector, Impact to Confidentiality) to generate a standardized score.</p>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 7: Vulnerability Reporting Engine */}
        <div ref={(el) => { slideRefs.current[6] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Vulnerability Reporting Engine</h1>
            <GlassCard className="space-y-6">
              <div className="content-box bg-white/5 p-6 rounded-xl flex items-start gap-4">
                <div className="p-2 bg-blue-500/20 rounded-lg">
                  <FileEdit className="w-8 h-8 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white mb-2">Standardized Intake</h3>
                  <p className="text-gray-300">A rigorous form designed to capture vulnerability Title, Category, and detailed technical descriptions.</p>
                </div>
              </div>
              <div className="content-box bg-white/5 p-6 rounded-xl flex items-start gap-4">
                <div className="p-2 bg-purple-500/20 rounded-lg">
                  <ImagePlus className="w-8 h-8 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white mb-2">Technical Evidence Support</h3>
                  <p className="text-gray-300">Full functionality for uploading various file types, specifically focusing on Proof of Concept (POC) videos and screenshots.</p>
                </div>
              </div>
              <div className="content-box bg-white/5 p-6 rounded-xl flex items-start gap-4">
                <div className="p-2 bg-green-500/20 rounded-lg">
                  <Tag className="w-8 h-8 text-green-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white mb-2">CWE Integration</h3>
                  <p className="text-gray-300">Standardized bug tagging using the Common Weakness Enumeration (CWE) database for technical consistency.</p>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 8: Unified Communication Engine (Chat) */}
        <div ref={(el) => { slideRefs.current[7] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Unified Communication Engine (Chat)</h1>
            <div className="space-y-4">
              <GlassCard className="content-box flex items-center space-x-4">
                <div className="p-3 bg-blue-500/20 rounded-full">
                  <MessageSquare className="w-8 h-8 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">Tripartite Thread</h3>
                  <p className="text-gray-300">A real-time chat interface integrated directly into every individual bug report.</p>
                </div>
              </GlassCard>
              <GlassCard className="content-box flex items-center space-x-4">
                <div className="p-3 bg-indigo-500/20 rounded-full">
                  <Send className="w-8 h-8 text-indigo-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">Researcher-Triager Interaction</h3>
                  <p className="text-gray-300">Direct channel for triagers to request further evidence or for researchers to clarify technical exploit steps.</p>
                </div>
              </GlassCard>
              <GlassCard className="content-box flex items-center space-x-4">
                <div className="p-3 bg-teal-500/20 rounded-full">
                  <Building2 className="w-8 h-8 text-teal-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">Company Engagement</h3>
                  <p className="text-gray-300">Organizations can join the existing thread to discuss remediation steps or assess the business impact of a finding.</p>
                </div>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 9: Triage & Validation Workflow */}
        <div ref={(el) => { slideRefs.current[8] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Triage & Validation Workflow</h1>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <GlassCard className="content-box flex flex-col items-center text-center">
                <div className="h-16 w-16 rounded-full bg-red-500/20 flex items-center justify-center mb-4">
                  <Lock className="w-8 h-8 text-red-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Collision Control</h3>
                <p className="text-gray-400">Implementation of a "Lock" mechanism that prevents multiple triagers from accessing or editing a report simultaneously.</p>
              </GlassCard>
              <GlassCard className="content-box flex flex-col items-center text-center">
                <div className="h-16 w-16 rounded-full bg-yellow-500/20 flex items-center justify-center mb-4">
                  <RefreshCw className="w-8 h-8 text-yellow-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Status Management</h3>
                <p className="text-gray-400">A robust state machine governing report transitions to statuses like Duplicate, Spam, Needs More Info, or Out-of-Scope.</p>
              </GlassCard>
              <GlassCard className="content-box flex flex-col items-center text-center">
                <div className="h-16 w-16 rounded-full bg-blue-500/20 flex items-center justify-center mb-4">
                  <FileText className="w-8 h-8 text-blue-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Technical Summaries</h3>
                <p className="text-gray-400">The ability for triagers to generate professional summaries for companies based on the validated researcher data.</p>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 10: Administrative Governance & Messaging */}
        <div ref={(el) => { slideRefs.current[9] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Administrative Governance & Messaging</h1>
            <GlassCard className="space-y-6">
              <div className="content-box border-l-4 border-blue-500 pl-4 py-2 flex items-center gap-4">
                <Megaphone className="w-8 h-8 text-blue-500 flex-shrink-0" />
                <div>
                  <h3 className="text-xl font-bold text-white">Global Broadcast System</h3>
                  <p className="text-gray-300">An administrative command center for sending high-priority messages to all platform users simultaneously.</p>
                </div>
              </div>
              <div className="content-box border-l-4 border-red-500 pl-4 py-2 flex items-center gap-4">
                <UserX className="w-8 h-8 text-red-500 flex-shrink-0" />
                <div>
                  <h3 className="text-xl font-bold text-white">User Moderation</h3>
                  <p className="text-gray-300">A dedicated interface for monitoring activity logs and suspending accounts that violate ethical guidelines.</p>
                </div>
              </div>
              <div className="content-box border-l-4 border-green-500 pl-4 py-2 flex items-center gap-4">
                <UserCog className="w-8 h-8 text-green-500 flex-shrink-0" />
                <div>
                  <h3 className="text-xl font-bold text-white">Staff Management</h3>
                  <p className="text-gray-300">Secure onboarding workflows for administrators to recruit and verify specialized technical triagers.</p>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 11: Company Program & Scoping */}
        <div ref={(el) => { slideRefs.current[10] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Company Program & Scoping</h1>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <GlassCard className="content-box text-center flex flex-col items-center">
                <Search className="w-10 h-10 text-white mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Program Directory</h3>
                <p className="text-gray-300">A searchable portal for researchers to discover public vulnerability disclosure programs.</p>
              </GlassCard>
              <GlassCard className="content-box text-center flex flex-col items-center">
                <Target className="w-10 h-10 text-white mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Scope Definition</h3>
                <p className="text-gray-300">Interface for companies to explicitly list "Out-of-Scope" assets to prevent unauthorized testing.</p>
              </GlassCard>
              <GlassCard className="content-box text-center flex flex-col items-center">
                <Activity className="w-10 h-10 text-white mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Engagement Dashboard</h3>
                <p className="text-gray-300">Companies can monitor the overall health of their programs and actively participate in triage chats.</p>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 12: Technical Architecture (Backend) */}
        <div ref={(el) => { slideRefs.current[11] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Technical Architecture (Backend)</h1>
            <GlassCard className="space-y-6">
              <div className="content-box flex items-start space-x-4">
                <div className="w-12 h-12 bg-green-500/20 rounded flex items-center justify-center flex-shrink-0">
                  <Server className="w-6 h-6 text-green-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">Platform Backbone</h3>
                  <p className="text-gray-300">High-performance API architecture built on Node.js and the Express.js framework.</p>
                </div>
              </div>
              <div className="content-box flex items-start space-x-4">
                <div className="w-12 h-12 bg-blue-500/20 rounded flex items-center justify-center flex-shrink-0">
                  <Shield className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">Security Layer</h3>
                  <p className="text-gray-300">Advanced bcrypt-based hashing for credentials and secure session token encryption.</p>
                </div>
              </div>
              <div className="content-box flex items-start space-x-4">
                <div className="w-12 h-12 bg-purple-500/20 rounded flex items-center justify-center flex-shrink-0">
                  <Database className="w-6 h-6 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">Data Layer</h3>
                  <p className="text-gray-300">MongoDB utilized for flexible, metadata-rich storage of vulnerability reports and user identities.</p>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 13: Technical Architecture (Frontend) */}
        <div ref={(el) => { slideRefs.current[12] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Technical Architecture (Frontend)</h1>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <GlassCard className="content-box text-center flex flex-col items-center">
                <Layers className="w-12 h-12 text-cyan-400 mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Interface Layer</h3>
                <p className="text-gray-300">React.js framework used to build highly responsive and real-time dashboards for all system actors.</p>
              </GlassCard>
              <GlassCard className="content-box text-center flex flex-col items-center">
                <Zap className="w-12 h-12 text-yellow-400 mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Real-time Updates</h3>
                <p className="text-gray-300">Integration of WebSocket or similar technologies to handle instant chat notifications and status changes.</p>
              </GlassCard>
              <GlassCard className="content-box text-center flex flex-col items-center">
                <Cpu className="w-12 h-12 text-pink-400 mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">CVSS Logic</h3>
                <p className="text-gray-300">Client-side implementation of CVSS scoring libraries for accurate, real-time severity calculations.</p>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 14: Use Case Demo: The "Verified Reporting" Lifecycle */}
        <div ref={(el) => { slideRefs.current[13] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">The "Verified Reporting" Lifecycle</h1>
            <div className="bg-white/5 rounded-2xl p-8 border border-white/10 backdrop-blur-sm">
              <div className="space-y-8 relative">
                {/* Step 1 */}
                <div className="content-box flex md:items-center flex-col md:flex-row gap-6 relative z-10">
                  <div className="w-12 h-12 rounded-full bg-blue-500 font-bold text-xl flex items-center justify-center shadow-[0_0_20px_rgba(59,130,246,0.5)]">
                    <UserCheck className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white">Researcher Registration</h3>
                    <p className="text-gray-300">Researcher registers, completes biometric KYC (NIC + Liveliness), and becomes a "Verified" contributor.</p>
                  </div>
                </div>
                {/* Step 2 */}
                <div className="content-box flex md:items-center flex-col md:flex-row gap-6 relative z-10">
                  <div className="w-12 h-12 rounded-full bg-purple-500 font-bold text-xl flex items-center justify-center shadow-[0_0_20px_rgba(168,85,247,0.5)]">
                    <Upload className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white">Submission</h3>
                    <p className="text-gray-300">Researcher uses the built-in CVSS tool and submits a bug report with video POC.</p>
                  </div>
                </div>
                {/* Step 3 */}
                <div className="content-box flex md:items-center flex-col md:flex-row gap-6 relative z-10">
                  <div className="w-12 h-12 rounded-full bg-emerald-500 font-bold text-xl flex items-center justify-center shadow-[0_0_20px_rgba(16,185,129,0.5)]">
                    <MessageCircle className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white">Triage & Coordination</h3>
                    <p className="text-gray-300">Triager locks the report, starts a chat with the researcher, and coordinates the impact assessment with the company.</p>
                  </div>
                </div>

                {/* Line connector */}
                <div className="absolute left-[23.5px] top-6 bottom-6 w-0.5 bg-gradient-to-b from-blue-500 via-purple-500 to-emerald-500 hidden md:block opacity-50"></div>
              </div>
            </div>
          </div>
        </div>

        {/* Slide 15: Conclusion & Milestone Roadmap */}
        <div ref={(el) => { slideRefs.current[14] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-4xl w-full text-center">
            <h1 className="text-4xl md:text-6xl font-bold text-white mb-8">Conclusion & Roadmap</h1>
            <GlassCard className="space-y-8 py-12">
              <div className="content-box">
                <h3 className="text-2xl font-medium text-white mb-2">Summary</h3>
                <p className="text-gray-400">Successfully implemented the core security, verification, and communication infrastructure of BugChase.</p>
              </div>
              <div className="w-32 h-px bg-white/20 mx-auto" />
              <div className="content-box">
                <h3 className="text-2xl font-medium text-white mb-2">Stability</h3>
                <p className="text-gray-400">The platform currently handles end-to-end report processing with high-level identity trust.</p>
              </div>
              <div className="w-32 h-px bg-white/20 mx-auto" />
              <div className="content-box">
                <h3 className="text-2xl font-medium text-blue-300">Next Milestone</h3>
                <p className="text-gray-400">Transitioning to Capstone 2, focusing on AI-driven duplicate detection, automated asset scanning, and local Escrow payment gateways.</p>
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