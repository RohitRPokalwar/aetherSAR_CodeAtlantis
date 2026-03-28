import React from 'react';
import { motion } from 'motion/react';
import { ArrowRight, Zap } from 'lucide-react';
import { useNavigate } from 'react-router';

const AetherFlowHero = () => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const navigate = useNavigate();

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let animationFrameId: number;
    let particles: Particle[] = [];
    const mouse = { x: null as number | null, y: null as number | null, radius: 200 };

    class Particle {
      x: number; y: number;
      directionX: number; directionY: number;
      size: number; color: string;
      constructor(x: number, y: number, directionX: number, directionY: number, size: number, color: string) {
        this.x = x; this.y = y;
        this.directionX = directionX; this.directionY = directionY;
        this.size = size; this.color = color;
      }
      draw() {
        ctx!.beginPath();
        ctx!.arc(this.x, this.y, this.size, 0, Math.PI * 2, false);
        ctx!.fillStyle = this.color;
        ctx!.fill();
      }
      update() {
        if (this.x > canvas!.width || this.x < 0) this.directionX = -this.directionX;
        if (this.y > canvas!.height || this.y < 0) this.directionY = -this.directionY;
        if (mouse.x !== null && mouse.y !== null) {
          let dx = mouse.x - this.x;
          let dy = mouse.y - this.y;
          let distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < mouse.radius + this.size) {
            const forceDirectionX = dx / distance;
            const forceDirectionY = dy / distance;
            const force = (mouse.radius - distance) / mouse.radius;
            this.x -= forceDirectionX * force * 5;
            this.y -= forceDirectionY * force * 5;
          }
        }
        this.x += this.directionX;
        this.y += this.directionY;
        this.draw();
      }
    }

    function init() {
      particles = [];
      const numberOfParticles = (canvas!.height * canvas!.width) / 9000;
      for (let i = 0; i < numberOfParticles; i++) {
        const size = Math.random() * 2 + 1;
        const x = Math.random() * ((innerWidth - size * 2) - size * 2) + size * 2;
        const y = Math.random() * ((innerHeight - size * 2) - size * 2) + size * 2;
        const directionX = Math.random() * 0.4 - 0.2;
        const directionY = Math.random() * 0.4 - 0.2;
        const color = 'rgba(0, 200, 255, 0.75)';
        particles.push(new Particle(x, y, directionX, directionY, size, color));
      }
    }

    const resizeCanvas = () => { canvas!.width = window.innerWidth; canvas!.height = window.innerHeight; init(); };
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    const connect = () => {
      for (let a = 0; a < particles.length; a++) {
        for (let b = a; b < particles.length; b++) {
          const distance = (particles[a].x - particles[b].x) ** 2 + (particles[a].y - particles[b].y) ** 2;
          if (distance < (canvas!.width / 7) * (canvas!.height / 7)) {
            const opacityValue = 1 - distance / 20000;
            const dx_a = particles[a].x - (mouse.x ?? 0);
            const dy_a = particles[a].y - (mouse.y ?? 0);
            const dist_a = Math.sqrt(dx_a * dx_a + dy_a * dy_a);
            ctx!.strokeStyle = mouse.x && dist_a < mouse.radius
              ? `rgba(255,255,255,${opacityValue})`
              : `rgba(0, 200, 255, ${opacityValue * 0.6})`;
            ctx!.lineWidth = 1;
            ctx!.beginPath();
            ctx!.moveTo(particles[a].x, particles[a].y);
            ctx!.lineTo(particles[b].x, particles[b].y);
            ctx!.stroke();
          }
        }
      }
    };

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      ctx!.fillStyle = '#060b14';
      ctx!.fillRect(0, 0, innerWidth, innerHeight);
      particles.forEach(p => p.update());
      connect();
    };

    const handleMouseMove = (e: MouseEvent) => { mouse.x = e.clientX; mouse.y = e.clientY; };
    const handleMouseOut = () => { mouse.x = null; mouse.y = null; };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseout', handleMouseOut);
    init();
    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseout', handleMouseOut);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  const fadeUpVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: (i: number) => ({
      opacity: 1, y: 0,
      transition: { delay: i * 0.2 + 0.5, duration: 0.8, ease: "easeInOut" },
    }),
  };

  return (
    <div className="relative h-screen w-full flex flex-col items-center justify-center overflow-hidden">
      <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full" />

      {/* Scanline overlay */}
      <div className="absolute inset-0 pointer-events-none"
        style={{ backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,200,255,0.015) 2px, rgba(0,200,255,0.015) 4px)' }} />

      <div className="relative z-10 text-center p-6">
        <motion.div custom={0} variants={fadeUpVariants} initial="hidden" animate="visible"
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-6 backdrop-blur-sm"
          style={{ background: 'rgba(0,200,255,0.08)', border: '1px solid rgba(0,200,255,0.2)' }}>
          <Zap className="h-4 w-4" style={{ color: '#00c8ff' }} />
          <span className="text-sm" style={{ color: '#d1d9e6' }}>Synthetic Aperture Radar · AI-Powered Detection</span>
        </motion.div>

        <motion.h1 custom={1} variants={fadeUpVariants} initial="hidden" animate="visible"
          className="text-5xl md:text-8xl font-black tracking-tighter mb-4"
          style={{ background: 'linear-gradient(to bottom, #ffffff, #00c8ff 60%, #0066aa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
          aetherSAR
        </motion.h1>

        <motion.div custom={1.5} variants={fadeUpVariants} initial="hidden" animate="visible"
          className="text-sm tracking-[0.4em] uppercase mb-6" style={{ color: '#00c8ff', opacity: 0.7 }}>
          Maritime Domain Awareness Platform
        </motion.div>

        <motion.p custom={2} variants={fadeUpVariants} initial="hidden" animate="visible"
          className="max-w-2xl mx-auto text-lg mb-10" style={{ color: '#b0bec5' }}>
          Military-grade ship detection from SAR satellite imagery using advanced AI inference pipelines, CFAR algorithms, and real-time GIS analytics.
        </motion.p>

        <motion.div custom={3} variants={fadeUpVariants} initial="hidden" animate="visible" className="flex items-center gap-4 justify-center">
          <button
            onClick={() => navigate('/dashboard')}
            className="px-8 py-4 rounded-lg flex items-center gap-3 transition-all duration-300"
            style={{ background: 'linear-gradient(135deg, #00c8ff, #0066cc)', color: '#ffffff', boxShadow: '0 0 30px rgba(0,200,255,0.3)' }}
            onMouseEnter={e => (e.currentTarget.style.boxShadow = '0 0 50px rgba(0,200,255,0.5)')}
            onMouseLeave={e => (e.currentTarget.style.boxShadow = '0 0 30px rgba(0,200,255,0.3)')}>
            Launch Platform
            <ArrowRight className="h-5 w-5" />
          </button>
          {/* <button className="px-8 py-4 rounded-lg border transition-all duration-300"
            style={{ borderColor: 'rgba(0,200,255,0.3)', color: '#94a3b8', background: 'transparent' }}>
            View Docs
          </button> */}
        </motion.div>

        {/* Stats row */}
        {/* <motion.div custom={4} variants={fadeUpVariants} initial="hidden" animate="visible"
          className="flex items-center gap-8 justify-center mt-16">
          {[['< 3s', 'Detection Latency'], ['99.2%', 'Precision'], ['2m GSD', 'Resolution'], ['2GB', 'Max Scene Size']].map(([val, label]) => (
            <div key={label} className="text-center">
              <div className="text-xl font-bold" style={{ color: '#00c8ff' }}>{val}</div>
              <div className="text-xs mt-1" style={{ color: '#475569' }}>{label}</div>
            </div>
          ))}
        </motion.div> */}
      </div>
    </div>
  );
};

export default AetherFlowHero;
