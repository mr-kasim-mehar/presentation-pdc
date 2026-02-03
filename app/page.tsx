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
            <GlassCard>
              <h1 className="text-3xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-600 mb-4 px-4 whitespace-nowrap">
                Parallel & Distributed Computing
              </h1>
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
                <p className="text-gray-300 text-lg">CUDA (Compute Unified Device Architecture) is a platform created by NVIDIA.</p>
              </div>
              <div className="content-box">
                <h3 className="text-xl font-semibold text-white mb-2">Purpose</h3>
                <p className="text-gray-300">It allows us to use the GPU (Graphics Processing Unit) for general-purpose mathematical calculations, not just for gaming.</p>
              </div>
              <div className="content-box">
                <h3 className="text-xl font-semibold text-white mb-2">Language</h3>
                <p className="text-gray-300">It uses an extension of the C language.</p>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 3: CPU vs. GPU */}
        <div ref={(el) => { slideRefs.current[2] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Why Parallelism Matters</h1>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <GlassCard className="content-box h-full">
                <div className="flex items-center gap-3 mb-4">
                  <Cpu className="w-8 h-8 text-blue-400" />
                  <h3 className="text-2xl font-bold text-white">CPU (The Manager)</h3>
                </div>
                <p className="text-gray-300 text-lg">Has a few powerful cores optimized for serial (one-by-one) tasks.</p>
                <div className="mt-6 p-4 bg-white/5 rounded-lg border border-white/10">
                  <p className="text-sm text-gray-400 italic">"A specialized professor"</p>
                </div>
              </GlassCard>
              <GlassCard className="content-box h-full">
                <div className="flex items-center gap-3 mb-4">
                  <Layers className="w-8 h-8 text-green-400" />
                  <h3 className="text-2xl font-bold text-white">GPU (The Workforce)</h3>
                </div>
                <p className="text-gray-300 text-lg">Has thousands of smaller cores optimized for parallel (simultaneous) tasks.</p>
                <div className="mt-6 p-4 bg-white/5 rounded-lg border border-white/10">
                  <p className="text-sm text-gray-400 italic">"A fleet of 1,000 bicycles"</p>
                </div>
              </GlassCard>
            </div>
          </div>
        </div>

        {/* Slide 4: Understanding the Hierarchy */}
        <div ref={(el) => { slideRefs.current[3] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Threads, Blocks, and Grids</h1>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <GlassCard className="content-box text-center">
                <div className="w-16 h-16 mx-auto bg-blue-500/20 rounded-full flex items-center justify-center mb-4">
                  <Zap className="w-8 h-8 text-blue-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Thread</h3>
                <p className="text-gray-300">The smallest unit that performs the addition.</p>
              </GlassCard>
              <GlassCard className="content-box text-center">
                <div className="w-16 h-16 mx-auto bg-purple-500/20 rounded-full flex items-center justify-center mb-4">
                  <Layers className="w-8 h-8 text-purple-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Block</h3>
                <p className="text-gray-300">A group of threads (e.g., 256 threads in one block).</p>
              </GlassCard>
              <GlassCard className="content-box text-center">
                <div className="w-16 h-16 mx-auto bg-orange-500/20 rounded-full flex items-center justify-center mb-4">
                  <LayoutDashboard className="w-8 h-8 text-orange-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Grid</h3>
                <p className="text-gray-300">A collection of blocks that handle the entire array.</p>
              </GlassCard>
            </div>
            <GlassCard className="mt-6 text-center">
              <p className="text-gray-300 italic">Logic: Every thread has a unique ID, so it knows which index of the array to add.</p>
            </GlassCard>
          </div>
        </div>

        {/* Slide 5: Memory Management */}
        <div ref={(el) => { slideRefs.current[4] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">Host vs. Device</h1>
            <GlassCard className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="content-box bg-white/5 p-6 rounded-xl text-center">
                  <h3 className="text-2xl font-bold text-blue-400 mb-2">Host</h3>
                  <p className="text-gray-300">Refers to the CPU and your system RAM.</p>
                </div>
                <div className="content-box bg-white/5 p-6 rounded-xl text-center">
                  <h3 className="text-2xl font-bold text-green-400 mb-2">Device</h3>
                  <p className="text-gray-300">Refers to the GPU and its specialized VRAM.</p>
                </div>
              </div>
              <div className="content-box bg-white/5 p-6 rounded-xl text-center border border-yellow-500/30">
                <h3 className="text-xl font-bold text-yellow-400 mb-2">Key Concept</h3>
                <p className="text-gray-300">The GPU cannot see the CPU's memory. We must copy data from the Host to the Device to process it.</p>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 6: The CUDA Workflow */}
        <div ref={(el) => { slideRefs.current[5] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">The 5-Step Process</h1>
            <GlassCard className="space-y-4">
              <div className="space-y-4">
                {[
                  "Allocate memory on the GPU (cudaMalloc).",
                  "Copy input arrays from CPU to GPU (cudaMemcpy).",
                  "Launch the Kernel (The GPU function).",
                  "Copy results from GPU back to CPU.",
                  "Free the memory to prevent leaks."
                ].map((step, idx) => (
                  <div key={idx} className="content-box flex items-center gap-4 p-4 bg-white/5 rounded-xl border border-white/10">
                    <div className="w-10 h-10 rounded-full bg-green-500/20 text-green-400 flex items-center justify-center font-bold text-xl flex-shrink-0">
                      {idx + 1}
                    </div>
                    <p className="text-lg text-gray-200">{step}</p>
                  </div>
                ))}
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Slide 7: Defining the Kernel */}
        <div ref={(el) => { slideRefs.current[6] = el; }} className="flex items-center justify-center min-h-screen px-4 snap-start snap-always pt-20">
          <div className="max-w-5xl w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-8 text-center">The GPU Function (__global__)</h1>
            <GlassCard>
              <p className="text-lg text-gray-300 mb-6 font-medium">We use the keyword <code className="text-green-400">__global__</code> to tell the compiler this function runs on the GPU.</p>
              <div className="bg-[#1e1e1e] p-6 rounded-xl border border-white/10 overflow-x-auto">
                <code className="text-sm md:text-base font-mono text-gray-300">
                  <span className="text-green-400">__global__</span> <span className="text-blue-400">void</span> <span className="text-yellow-300">addArrays</span>(<span className="text-blue-400">int</span> *a, <span className="text-blue-400">int</span> *b, <span className="text-blue-400">int</span> *c, <span className="text-blue-400">int</span> n) {"{"}<br />
                  &nbsp;&nbsp;<span className="text-gray-500">// Find the unique index for this thread</span><br />
                  &nbsp;&nbsp;<span className="text-blue-400">int</span> i = blockIdx.x * blockDim.x + threadIdx.x;<br />
                  <br />
                  &nbsp;&nbsp;<span className="text-purple-400">if</span> (i &lt; n) {"{"}<br />
                  &nbsp;&nbsp;&nbsp;&nbsp;c[i] = a[i] + b[i]; <span className="text-gray-500">// Addition happens here!</span><br />
                  &nbsp;&nbsp;{"}"}<br />
                  {"}"}
                </code>
              </div>
            </GlassCard>
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
                <div className="h-px w-32 bg-white/20 mx-auto my-6" />
                <h3 className="text-2xl font-bold text-white mb-4">Q&A</h3>
                <p className="text-gray-300 mb-6">We are now open for any questions.</p>
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