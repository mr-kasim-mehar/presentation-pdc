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
  { id: 0, title: "Title", name: "Introduction to Parallel Computing" },
  { id: 1, title: "Basics", name: "What is CUDA?" },
  { id: 2, title: "CPU/GPU", name: "Why Parallelism Matters" },
  { id: 3, title: "Grid", name: "Threads, Blocks, and Grids" },
  { id: 4, title: "Memory", name: "Host vs. Device" },
  { id: 5, title: "Steps", name: "The CUDA Workflow" },
  { id: 6, title: "Kernel", name: "Defining the Kernel" },
  { id: 7, title: "Alloc", name: "Allocating Memory" },
  { id: 8, title: "Copy", name: "Moving the Data" },
  { id: 9, title: "Launch", name: "Launching the Kernel" },
  { id: 10, title: "Code", name: "Full Code Implementation" },
  { id: 11, title: "End", name: "Thank You!" },
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



        {/* Slide 1: Title Slide */}
        <div ref={(el) => { slideRefs.current[0] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full text-center space-y-8">
            <h1 className="text-3xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-600 mb-8 px-4 whitespace-nowrap">
              Parallel & Distributed Computing
            </h1>
            <GlassCard>
              <p className="text-xl md:text-2xl text-green-200 font-medium mb-6">
                What is CUDA and How to Write Code?
              </p>
              <div className="text-lg text-gray-300 mt-4 space-x-4">
                <span className="inline-block">Danish Ali</span>
                <span className="inline-block">Abdullah Azam</span>
                <span className="inline-block">Ameer Hamza Bajwa</span>
                <span className="inline-block">Muhammad Qasim</span>
              </div>
              <p className="text-gray-400 mt-2 italic">Gift University, Gujranwala</p>
            </GlassCard>
          </div>
        </div>

        {/* Slide 2: What is CUDA? */}
        <div ref={(el) => { slideRefs.current[1] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">The Basics</h1>
            <GlassCard className="space-y-6">
              <div className="content-box">
                <h3 className="text-2xl font-bold text-green-400 mb-2">Definition</h3>
                <p className="text-gray-300 text-lg">CUDA (Compute Unified Device Architecture) is a platform created by NVIDIA that allows software to use the GPU for general purpose processing.</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                <div className="content-box bg-white/5 p-4 rounded-lg">
                  <h4 className="text-xl font-semibold text-blue-300 mb-2">The Concept</h4>
                  <p className="text-gray-300">Unlocks the power of the graphics card for math, science, and AI, not just gaming.</p>
                </div>
                <div className="content-box bg-white/5 p-4 rounded-lg">
                  <h4 className="text-xl font-semibold text-purple-300 mb-2">The Analogy</h4>
                  <p className="text-gray-300">
                    Think of your <strong>CPU</strong> as a brilliant <span className="text-blue-200">Math Professor</span> (smart, but works alone).
                    <br /><br />
                    Think of <strong>CUDA</strong> as hiring <span className="text-purple-200">1,000 Students</span> (less experienced, but they work together to finish the job much faster).
                  </p>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 3: CPU vs. GPU */}
        <div ref={(el) => { slideRefs.current[2] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Why Parallelism Matters</h1>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <GlassCard className="content-box h-full flex flex-col">
                <div className="flex items-center gap-3 mb-4">
                  <Cpu className="w-8 h-8 text-blue-400" />
                  <h3 className="text-2xl font-bold text-white">CPU (The Manager)</h3>
                </div>
                <p className="text-gray-300 text-lg mb-4">Few powerful cores. Optimized for serial processing (doing one thing at a time very quickly).</p>
                <div className="mt-auto p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
                  <h4 className="text-blue-300 font-bold mb-2">Like a Race Car</h4>
                  <p className="text-sm text-gray-300">Extremely fast for one person, but can't move 50 people at once.</p>
                </div>
              </GlassCard>
              <GlassCard className="content-box h-full flex flex-col">
                <div className="flex items-center gap-3 mb-4">
                  <Layers className="w-8 h-8 text-green-400" />
                  <h3 className="text-2xl font-bold text-white">GPU (The Workforce)</h3>
                </div>
                <p className="text-gray-300 text-lg mb-4">Thousands of smaller cores. Optimized for parallel processing (doing many things at once).</p>
                <div className="mt-auto p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                  <h4 className="text-green-300 font-bold mb-2">Like a City Bus</h4>
                  <p className="text-sm text-gray-300">Slower top speed than a race car, but transports 50 people simultaneously.</p>
                </div>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 4: Understanding the Hierarchy */}
        <div ref={(el) => { slideRefs.current[3] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Threads, Blocks, and Grids</h1>
            <p className="text-center text-gray-300 mb-8 max-w-2xl mx-auto">Understanding how CUDA organizes work using a <span className="text-orange-400 font-bold">Construction Site</span> analogy.</p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <GlassCard className="content-box text-center">
                <div className="w-16 h-16 mx-auto bg-blue-500/20 rounded-full flex items-center justify-center mb-4">
                  <Zap className="w-8 h-8 text-blue-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Thread</h3>
                <p className="text-blue-200 text-sm font-semibold mb-2">"The Worker"</p>
                <p className="text-gray-300">One worker laying a single brick. The smallest unit of execution.</p>
              </GlassCard>
              <GlassCard className="content-box text-center">
                <div className="w-16 h-16 mx-auto bg-purple-500/20 rounded-full flex items-center justify-center mb-4">
                  <Layers className="w-8 h-8 text-purple-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Block</h3>
                <p className="text-purple-200 text-sm font-semibold mb-2">"The Team"</p>
                <p className="text-gray-300">A group of workers building one wall together. Threads in a block can share memory.</p>
              </GlassCard>
              <GlassCard className="content-box text-center">
                <div className="w-16 h-16 mx-auto bg-orange-500/20 rounded-full flex items-center justify-center mb-4">
                  <LayoutDashboard className="w-8 h-8 text-orange-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Grid</h3>
                <p className="text-orange-200 text-sm font-semibold mb-2">"The Site"</p>
                <p className="text-gray-300">The entire construction site. A collection of all blocks working on the full problem.</p>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 5: Memory Management */}
        <div ref={(el) => { slideRefs.current[4] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Host vs. Device</h1>
            <GlassCard className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="content-box bg-white/5 p-6 rounded-xl text-center">
                  <h3 className="text-2xl font-bold text-blue-400 mb-2">Host (CPU)</h3>
                  <p className="text-gray-300">System RAM. Where your main program starts.</p>
                </div>
                <div className="content-box bg-white/5 p-6 rounded-xl text-center">
                  <h3 className="text-2xl font-bold text-green-400 mb-2">Device (GPU)</h3>
                  <p className="text-gray-300">Video RAM (VRAM). Where the heavy lifting happens.</p>
                </div>
              </div>
              <div className="content-box bg-white/5 p-6 rounded-xl border border-yellow-500/30 flex flex-col md:flex-row items-center gap-6">
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-yellow-400 mb-2">The Bottleneck: Data Travel</h3>
                  <p className="text-gray-300">The GPU cannot access CPU memory directly. Moving data is the slowest part.</p>
                </div>
                <div className="flex-1 bg-black/20 p-4 rounded-lg">
                  <p className="text-gray-300 italic text-sm">
                    "Processing on the GPU is instant, but getting data there is like <strong>shipping a package</strong>. You want to ship a full truckload (large data), not just one envelope at a time."
                  </p>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 6: The CUDA Workflow */}
        <div ref={(el) => { slideRefs.current[5] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">The CUDA Workflow</h1>
            <GlassCard className="space-y-6">
              <p className="text-center text-gray-300 text-lg mb-4">Think of it like a <span className="text-green-400 font-bold">Chef's Workflow</span>.</p>
              <div className="space-y-3">
                {[
                  { title: "Allocate", desc: "Get bowls ready (Reserve GPU Memory)", icon: "🥣" },
                  { title: "Copy", desc: "Pour ingredients into bowls (Send Data CPU → GPU)", icon: "🥛" },
                  { title: "Launch", desc: "Turn on the mixer (Execute Kernel on GPU)", icon: "⚙️" },
                  { title: "Copy Back", desc: "Pour cake batter back into pan (Send Results GPU → CPU)", icon: "🎂" },
                  { title: "Free", desc: "Wash the bowls (Free GPU Memory)", icon: "🧼" }
                ].map((step, idx) => (
                  <div key={idx} className="content-box flex items-center gap-4 p-4 bg-white/5 rounded-xl border border-white/10 hover:bg-white/10 transition-colors">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-green-500 to-emerald-700 text-white flex items-center justify-center font-bold text-xl flex-shrink-0 shadow-lg">
                      {idx + 1}
                    </div>
                    <div className="flex-1">
                      <h4 className="text-white font-bold text-lg">{step.title}</h4>
                      <p className="text-gray-300">{step.desc}</p>
                    </div>
                    <div className="text-2xl grayscale opacity-50 hover:grayscale-0 hover:opacity-100 transition-all">
                      {step.icon}
                    </div>
                  </div>
                ))}
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 7: Defining the Kernel */}
        <div ref={(el) => { slideRefs.current[6] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-6xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">The GPU Function (__global__)</h1>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <GlassCard>
                <div className="bg-[#1e1e1e] p-6 rounded-xl border border-white/10 overflow-x-auto h-full flex flex-col justify-center">
                  <code className="text-sm md:text-base font-mono text-gray-300">
                    <span className="text-green-400">__global__</span> <span className="text-blue-400">void</span> <span className="text-yellow-300">addArrays</span>(...) {"{"}<br /><br />
                    &nbsp;&nbsp;<span className="text-gray-500">// Calculate unique ID</span><br />
                    &nbsp;&nbsp;<span className="text-blue-400">int</span> i = blockIdx.x * blockDim.x + threadIdx.x;<br />
                    <br />
                    &nbsp;&nbsp;<span className="text-purple-400">if</span> (i &lt; n) {"{"}<br />
                    &nbsp;&nbsp;&nbsp;&nbsp;c[i] = a[i] + b[i];<br />
                    &nbsp;&nbsp;{"}"}<br />
                    {"}"}
                  </code>
                </div>
              </GlassCard>

              <GlassCard className="space-y-4">
                <h3 className="text-xl font-bold text-white border-b border-white/10 pb-2">Translation</h3>
                <div className="space-y-4">
                  <div className="bg-white/5 p-3 rounded-lg">
                    <code className="text-green-400 font-bold block mb-1">__global__</code>
                    <p className="text-gray-300 text-sm">"Hey Compiler, this function is special. It runs on the GPU and is called from the CPU."</p>
                  </div>
                  <div className="bg-white/5 p-3 rounded-lg">
                    <code className="text-blue-400 font-bold block mb-1">blockIdx.x * blockDim.x</code>
                    <p className="text-gray-300 text-sm">"Which team (Block) am I in, and how big is that team?"</p>
                  </div>
                  <div className="bg-white/5 p-3 rounded-lg">
                    <code className="text-yellow-300 font-bold block mb-1">+ threadIdx.x</code>
                    <p className="text-gray-300 text-sm">"Which worker number am I inside my team?"</p>
                  </div>
                  <div className="bg-white/5 p-3 rounded-lg border-l-4 border-green-500">
                    <p className="text-white text-sm font-bold">i = Global ID</p>
                    <p className="text-gray-300 text-sm">Calculating 'i' gives every thread a unique ID badge so it knows exactly which number in the array to process.</p>
                  </div>
                </div>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 8: Allocating Memory */}
        <div ref={(el) => { slideRefs.current[7] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Step 1: cudaMalloc</h1>
            <GlassCard>
              <p className="text-lg text-gray-300 mb-6 font-medium">Just like <code className="text-orange-400">malloc()</code> in standard C, we use <code className="text-green-400">cudaMalloc</code> for the GPU.</p>
              <div className="bg-[#1e1e1e] p-6 rounded-xl border border-white/10 overflow-x-auto">
                <code className="text-sm md:text-base font-mono text-gray-300">
                  <span className="text-blue-400">int</span> *d_a, *d_b, *d_c; <span className="text-gray-500">// 'd' stands for Device</span><br />
                  <span className="text-blue-400">int</span> size = n * <span className="text-purple-400">sizeof</span>(<span className="text-blue-400">int</span>);<br />
                  <br />
                  <span className="text-yellow-300">cudaMalloc</span>(&d_a, size);<br />
                  <span className="text-yellow-300">cudaMalloc</span>(&d_b, size);<br />
                  <span className="text-yellow-300">cudaMalloc</span>(&d_c, size);
                </code>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 9: Moving the Data */}
        <div ref={(el) => { slideRefs.current[8] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Step 2: cudaMemcpy</h1>
            <GlassCard>
              <p className="text-lg text-gray-300 mb-6 font-medium">We move the numbers from the RAM to the Graphics Card.</p>
              <div className="bg-[#1e1e1e] p-6 rounded-xl border border-white/10 overflow-x-auto">
                <code className="text-sm md:text-base font-mono text-gray-300">
                  <span className="text-gray-500">// From Host (CPU) to Device (GPU)</span><br />
                  <span className="text-yellow-300">cudaMemcpy</span>(d_a, h_a, size, cudaMemcpyHostToDevice);<br />
                  <span className="text-yellow-300">cudaMemcpy</span>(d_b, h_b, size, cudaMemcpyHostToDevice);
                </code>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 10: Launching the Kernel */}
        <div ref={(el) => { slideRefs.current[9] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Step 3: The &lt;&lt;&lt; &gt;&gt;&gt; Syntax</h1>
            <GlassCard>
              <p className="text-lg text-gray-300 mb-6 font-medium">The triple angle brackets tell the GPU how many blocks and threads to use.</p>
              <div className="bg-[#1e1e1e] p-6 rounded-xl border border-white/10 overflow-x-auto">
                <code className="text-sm md:text-base font-mono text-gray-300">
                  <span className="text-blue-400">int</span> threadsPerBlock = 256;<br />
                  <span className="text-blue-400">int</span> blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;<br />
                  <br />
                  <span className="text-yellow-300">addArrays</span>&lt;&lt;&lt;blocksPerGrid, threadsPerBlock&gt;&gt;&gt;(d_a, d_b, d_c, n);
                </code>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 11: Full Code Implementation */}
        <div ref={(el) => { slideRefs.current[10] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-6xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-6 text-center">Complete Source Code (main.cu)</h1>
            <GlassCard className="overflow-y-auto max-h-[70vh]">
              <div className="bg-[#1e1e1e] p-6 rounded-xl border border-white/10 overflow-x-auto">
                <code className="text-sm font-mono text-gray-300 leading-relaxed">
                  <span className="text-purple-400">#include</span> &lt;stdio.h&gt;<br /><br />
                  <span className="text-green-400">__global__</span> <span className="text-blue-400">void</span> <span className="text-yellow-300">add</span>(<span className="text-blue-400">int</span> *a, <span className="text-blue-400">int</span> *b, <span className="text-blue-400">int</span> *c, <span className="text-blue-400">int</span> n) {"{"}<br />
                  &nbsp;&nbsp;<span className="text-blue-400">int</span> i = blockIdx.x * blockDim.x + threadIdx.x;<br />
                  &nbsp;&nbsp;<span className="text-purple-400">if</span> (i &lt; n) c[i] = a[i] + b[i];<br />
                  {"}"}<br /><br />
                  <span className="text-blue-400">int</span> <span className="text-yellow-300">main</span>() {"{"}<br />
                  &nbsp;&nbsp;<span className="text-blue-400">int</span> n = 10;<br />
                  &nbsp;&nbsp;<span className="text-blue-400">int</span> size = n * <span className="text-purple-400">sizeof</span>(<span className="text-blue-400">int</span>);<br />
                  &nbsp;&nbsp;<span className="text-blue-400">int</span> h_a[10] = {"{1,2,3,4,5,6,7,8,9,10}"}, h_b[10] = {"{1,1,1,1,1,1,1,1,1,1}"}, h_c[10];<br />
                  &nbsp;&nbsp;<span className="text-blue-400">int</span> *d_a, *d_b, *d_c;<br /><br />
                  &nbsp;&nbsp;<span className="text-yellow-300">cudaMalloc</span>(&d_a, size); <span className="text-yellow-300">cudaMalloc</span>(&d_b, size); <span className="text-yellow-300">cudaMalloc</span>(&d_c, size);<br />
                  &nbsp;&nbsp;<span className="text-yellow-300">cudaMemcpy</span>(d_a, h_a, size, cudaMemcpyHostToDevice);<br />
                  &nbsp;&nbsp;<span className="text-yellow-300">cudaMemcpy</span>(d_b, h_b, size, cudaMemcpyHostToDevice);<br /><br />
                  &nbsp;&nbsp;<span className="text-yellow-300">add</span>&lt;&lt;&lt;1, n&gt;&gt;&gt;(d_a, d_b, d_c, n);<br /><br />
                  &nbsp;&nbsp;<span className="text-yellow-300">cudaMemcpy</span>(h_c, d_c, size, cudaMemcpyDeviceToHost);<br />
                  &nbsp;&nbsp;<span className="text-purple-400">for</span>(<span className="text-blue-400">int</span> i=0; i&lt;n; i++) <span className="text-yellow-300">printf</span>(<span className="text-orange-400">"%d + %d = %d\n"</span>, h_a[i], h_b[i], h_c[i]);<br /><br />
                  &nbsp;&nbsp;<span className="text-yellow-300">cudaFree</span>(d_a); <span className="text-yellow-300">cudaFree</span>(d_b); <span className="text-yellow-300">cudaFree</span>(d_c);<br />
                  &nbsp;&nbsp;<span className="text-purple-400">return</span> 0;<br />
                  {"}"}
                </code>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 12: Thank You! */}
        <div ref={(el) => { slideRefs.current[11] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-4xl w-full text-center">
            <h1 className="text-4xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-teal-500 mb-8">
              Conclusion & Questions
            </h1>
            <GlassCard className="space-y-8 py-12">
              <div className="content-box">
                <h3 className="text-2xl font-bold text-white mb-4">Summary</h3>
                <p className="text-gray-300 mb-6">CUDA unlocks the massive power of GPUs for everyday tasks.</p>

                <div className="text-gray-400 italic space-y-1">
                  <p>Contact:</p>
                  <p>Danish Ali, Abdullah Azam</p>
                  <p>Ameer Hamza Bajwa, Muhammad Qasim</p>
                </div>
                <p className="text-green-400 font-bold mt-8 text-xl">Thank You for your time!</p>
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